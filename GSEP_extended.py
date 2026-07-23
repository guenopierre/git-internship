#%% sys

from __future__ import annotations

import sys
import pandas as pd
import numpy as np


sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

from usefull_functions import time_mean, convert_prefix_value
import dataset_reading

#%% merge helper functions (unchanged)

def merge_ar_info(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge AR (Active Region) info from df2 into df1 based on:
      1. Same calendar date between df1['fl_start_time'] and df2['DATETIME']
      2. df2['AR_number'] + k*10000 == df1['noaa_ar'], for k in (0, 1)

    If exactly one row of df2 survives the filtering, its info is copied
    into new columns of df1:
        Location  -> AR_location
        Lo        -> AR_lo
        Area      -> AR_area
        Z         -> AR_z
        LL        -> AR_ll
        NN        -> AR_nn
        Mag_type  -> AR_mag_type

    If zero rows match, the new columns are left as None for that row.
    If more than one row matches (ambiguous), a warning is printed and
    the first match is used (adjust the `matched.iloc[0]` line below if
    you'd rather handle ambiguity differently).
    """

    df1 = df1.copy()
    df2 = df2.copy()

    # --- Ensure proper datetime dtypes ---
    df1['fl_start_time'] = pd.to_datetime(df1['fl_start_time'])
    df2['DATETIME'] = pd.to_datetime(df2['DATETIME'])

    # Precompute date-only column in df2 once, for fast filtering
    df2['_date_only'] = df2['DATETIME'].dt.date

    # New columns to fill in df1
    new_cols = ['AR_location', 'AR_lo', 'AR_area', 'AR_z',
                'AR_ll', 'AR_nn', 'AR_mag_type']
    for col in new_cols:
        df1[col] = None

    total = len(df1)

    # Use positional enumeration so the progress counter is always 1..total,
    # regardless of df1's actual index values.
    for pos, (idx, row) in enumerate(df1.iterrows(), start=1):
        print(f"line {pos} over {total}")

        fl_date = row['fl_start_time'].date()
        noaa_ar = row['noaa_ar']

        # --- Step 2: same date filter ---
        candidates = df2[df2['_date_only'] == fl_date]
        if candidates.empty:
            continue

        # --- Step 3: AR_number + k*10000 == noaa_ar, k = 0 or 1 ---
        mask = (candidates['AR_number'] == noaa_ar) | \
               ((candidates['AR_number'] + 10000) == noaa_ar)
        matched = candidates[mask]

        if matched.empty:
            continue

        if len(matched) > 1:
            print(f"  -> WARNING: {len(matched)} ambiguous matches for "
                  f"df1 row {idx} (date={fl_date}, noaa_ar={noaa_ar}). "
                  f"Using the first match.")

        match_row = matched.iloc[0]

        # --- Step 4: copy info over ---
        df1.at[idx, 'AR_location'] = match_row['Location']
        df1.at[idx, 'AR_lo'] = match_row['Lo']
        df1.at[idx, 'AR_area'] = match_row['Area']
        df1.at[idx, 'AR_z'] = match_row['Z']
        df1.at[idx, 'AR_ll'] = match_row['LL']
        df1.at[idx, 'AR_nn'] = match_row['NN']
        df1.at[idx, 'AR_mag_type'] = match_row['Mag_type']

    return df1


def merge_daily_sn(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sunspot number info from df2 into df1 based on:
      Same calendar date between df1['fl_start_time'] and df2['datetime']

    df2['datetime'] is built from its 'year', 'month', 'day' columns,
    in the format YYYY-MM-DD 00:00:01.

    If exactly one row of df2 survives the date filtering, its
    'daily_total_sn' value is copied into a new 'daily_sn' column of df1.

    If zero rows match, 'daily_sn' is left as None for that row.
    If more than one row matches (ambiguous), a warning is printed and
    the first match is used.
    """

    df1 = df1.copy()
    df2 = df2.copy()

    # --- Build the datetime column in df2 from year/month/day ---
    df2['datetime'] = pd.to_datetime(
        dict(year=df2['year'], month=df2['month'], day=df2['day'])
    ) + pd.Timedelta(seconds=1)  # -> YYYY-MM-DD 00:00:01

    # --- Ensure proper datetime dtype in df1 ---
    df1['fl_start_time'] = pd.to_datetime(df1['fl_start_time'])

    # Precompute date-only column in df2 once, for fast filtering
    df2['_date_only'] = df2['datetime'].dt.date

    # New column to fill in df1
    df1['daily_sn'] = None

    total = len(df1)

    for pos, (idx, row) in enumerate(df1.iterrows(), start=1):
        print(f"line {pos} over {total}")

        fl_date = row['fl_start_time'].date()

        # --- Same date filter ---
        matched = df2[df2['_date_only'] == fl_date]

        if matched.empty:
            continue

        if len(matched) > 1:
            print(f"  -> WARNING: {len(matched)} ambiguous matches for "
                  f"df1 row {idx} (date={fl_date}). Using the first match.")

        match_row = matched.iloc[0]

        # --- Copy info over ---
        df1.at[idx, 'daily_sn'] = match_row['daily_total_sn']

    return df1

#%% one function per feature ("flag")

def add_xray_flux(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'fl_goes_xray' (numeric GOES X-ray class, from 'fl_goes_class')."""
    df = df.copy()
    df['fl_goes_xray'] = df['fl_goes_class'].apply(convert_prefix_value)
    return df


def add_ar_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge NOAA Active Region info, map it to integer codes, merge the
    Zurich classification lookup table, and rank-encode the resulting
    categorical columns against 'noaa_pf10MeV'.
    """
    df = df.copy()
    srs_combine_complete = dataset_reading.load_srs_combine_complete()

    mapping = {
        'ALPHA': 1,
        'BETA': 2,
        'GAMMA': 3,
        'BETA-GAMMA': 4,
        'DELTA': 5,
        'BETA-DELTA': 6,
        'BETA-GAMMA-DELTA': 7,
        'GAMMA-DELTA': 8
    }

    mapping2 = {
        "AXX": 1, "BXO": 2, "BXI": 3, "HRX": 4, "CRO": 5, "CRI": 6,
        "HAX": 7, "CAO": 8, "CAI": 9, "HSX": 10, "CSO": 11, "CSI": 12,
        "DRO": 13, "ERO": 14, "FRO": 15, "DRI": 16, "ERI": 17, "FRI": 18,
        "DAO": 19, "EAO": 20, "FAO": 21, "DAI": 22, "EAI": 23, "FAI": 24,
        "DSO": 25, "ESO": 26, "FSO": 27, "DSI": 28, "ESI": 29, "FSI": 30,
        "DAC": 31, "EAC": 32, "FAC": 33, "DSC": 34, "ESC": 35, "FSC": 36,
        "HKX": 37, "CKO": 38, "CKI": 39, "HHX": 40, "CHO": 41, "CHI": 42,
        "DKO": 43, "EKO": 44, "FKO": 45, "DKI": 46, "EKI": 47, "FKI": 48,
        "DHO": 49, "EHO": 50, "FHO": 51, "DHI": 52, "EHI": 53, "FHI": 54,
        "DKC": 55, "EKC": 56, "FKC": 57, "DHC": 58, "EHC": 59, "FHC": 60
    }

    mapping_letter1 = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "H": 7}
    mapping_letter2 = {"X": 1, "R": 2, "S": 3, "A": 4, "H": 5, "K": 6}
    mapping_letter3 = {"X": 1, "O": 2, "I": 3, "C": 4}

    df = merge_ar_info(df, srs_combine_complete)  # own function

    df['AR_mag_type'] = df['AR_mag_type'].str.upper()  # put str in CAPITAL
    df['AR_mag_type_int'] = df['AR_mag_type'].map(mapping)

    df['AR_z'] = df['AR_z'].str.upper()
    df['AR_z_int'] = df['AR_z'].map(mapping2)

    df['group_configuration'] = df['AR_z'].str[0]
    df['largest_spot_type'] = df['AR_z'].str[1]
    df['spots_distribution'] = df['AR_z'].str[2]

    df['group_configuration_int'] = df['group_configuration'].map(mapping_letter1)
    df['largest_spot_type_int'] = df['largest_spot_type'].map(mapping_letter2)
    df['spots_distribution_int'] = df['spots_distribution'].map(mapping_letter3)

    # 1. Load the lookup table from the sheet provided
    lookup = pd.read_excel(
        "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/zurich classification parameters.xlsx",
        sheet_name="Sheet1",
        usecols=["ZMcI-type", "Magnetic type", "Length", "Penumbra type", "Distribution"]
    )

    # 2. Rename lookup columns to match desired output names,
    #    and rename the key column to match df's column name (AR_z)
    lookup = lookup.rename(columns={
        "ZMcI-type": "AR_z",
        "Magnetic type": "AR_z_magnetic_type",
        "Length": "AR_z_length",
        "Penumbra type": "AR_z_penumbra_type",
        "Distribution": "AR_z_distribution",
    })

    lookup["AR_z"] = lookup["AR_z"].str.upper()

    # 3. Merge on AR_z (left join keeps all rows of df, including NaNs / unmatched codes)
    df = df.merge(lookup, on="AR_z", how="left")

    cols_to_rank = [
        'AR_z',
        'AR_mag_type',
        'group_configuration',
        'largest_spot_type',
        'spots_distribution',
        'AR_z_magnetic_type',
        'AR_z_length',
        'AR_z_penumbra_type',
        'AR_z_distribution',
    ]

    for col in cols_to_rank:
        sorted_categories = df.groupby(col)['noaa_pf10MeV'].mean().sort_values().index
        new_mapping = {cat: rank for rank, cat in enumerate(sorted_categories, start=1)}
        df[f'{col}_int_ranked'] = df[col].map(new_mapping)

    return df


def add_sunspot_number(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the daily total sunspot number into 'daily_sn'."""
    df = df.copy()

    SN_d_tot_V2 = pd.read_csv(
        "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/sunspot/SN_d_tot_V2.0.csv",
        delimiter=";",
        header=None,
        names=['year', 'month', 'day', 'date_decimal', 'daily_total_sn',
               'daily_std_variation', 'nb_observations', 'definitive/provisional']
    )

    df = merge_daily_sn(df, SN_d_tot_V2)

    del SN_d_tot_V2
    return df


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add CME / flare / radio-burst flags and the SEP intensity classes."""
    df = df.copy()

    df['CME flag'] = df['cme_id'].notna().astype(int)

    df['flare flag'] = df['fl_id'].notna().astype(int)

    df['CME + flare flag'] = (df['cme_id'].notna() & df['fl_id'].notna()).astype(int)

    df['radio burst 1'] = df['gsep_notes'].str.startswith('Type II Radio burst ').astype(int)  # (from 'gsep_notes')
    df['radio burst 2'] = df['m_type2_onset_time'].notna().astype(int)  # (from 'm_type2_onset_time')

    df['>= S1'] = (df['noaa_pf10MeV'] > 10).astype(int)
    df['>= S2'] = (df['noaa_pf10MeV'] > 100).astype(int)
    df['>= S3'] = (df['noaa_pf10MeV'] > 1000).astype(int)

    df['= S1'] = ((df['noaa_pf10MeV'] >= 10) & (df['noaa_pf10MeV'] < 100)).astype(int)
    df['= S2'] = ((df['noaa_pf10MeV'] >= 100) & (df['noaa_pf10MeV'] < 1000)).astype(int)
    df['= S3'] = ((df['noaa_pf10MeV'] >= 1000) & (df['noaa_pf10MeV'] < 10000)).astype(int)
    df['= S4'] = (df['noaa_pf10MeV'] > 10000).astype(int)

    conditions = [
        (df['noaa_pf10MeV'] >= 10) & (df['noaa_pf10MeV'] < 100),
        (df['noaa_pf10MeV'] >= 100) & (df['noaa_pf10MeV'] < 1000),
        (df['noaa_pf10MeV'] >= 1000) & (df['noaa_pf10MeV'] < 10000),
        (df['noaa_pf10MeV'] > 10000),
    ]
    valeurs = [1, 2, 3, 4]

    df['S_class'] = np.select(conditions, valeurs, default=0)

    return df


def add_slice_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'slice range': duration in minutes between slice_start and slice_end."""
    df = df.copy()
    _, df_clean, _ = time_mean(df['slice_start'], df['slice_end'], diff_max=1000)
    diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(df.index)
    df['slice range'] = diff_minutes
    return df


def add_ref1(df: pd.DataFrame) -> pd.DataFrame:
    """Add time differences (minutes) of several reference times vs 'timestamp'."""
    df = df.copy()
    for col, out_name in [
        ('cdaw_start_time', 'cdaw_start_time ref1'),
        ('cme_launch_time', 'cme_launch_time ref1'),
        ('fl_start_time', 'fl_start_time ref1'),
        ('fl_peak_time', 'fl_peak_time ref1'),
    ]:
        _, df_clean, _ = time_mean(df[col], df['timestamp'], diff_max=1000)
        diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(df.index)
        df[out_name] = diff_minutes
    return df


def add_ref2(df: pd.DataFrame) -> pd.DataFrame:
    """Add time differences (minutes) of several reference times vs 'fl_start_time'."""
    df = df.copy()
    for col, out_name in [
        ('cdaw_start_time', 'cdaw_start_time ref2'),
        ('cme_launch_time', 'cme_launch_time ref2'),
        ('fl_peak_time', 'fl_peak_time ref2'),
        ('timestamp', 'timestamp ref2'),
    ]:
        _, df_clean, _ = time_mean(df[col], df['fl_start_time'], diff_max=1000)
        diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(df.index)
        df[out_name] = diff_minutes
    return df


def convert_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force numeric dtype (pd.to_numeric, invalid -> NaN) on the columns used
    for modelling. Columns coming from a disabled feature (e.g. 'daily_sn'
    if add_sunspot_number was not applied) are silently skipped.
    """
    df = df.copy()
    cols_to_numeric = [
        'p_cme_width',
        'p_cme_speed',
        'Inst_category',
        'noaa_pf10MeV',
        'noaa-sep_flag',
        'radio burst 1',
        'radio burst 2',
        'AR_mag_type_int',
        'AR_area',
        'daily_sn',
    ]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


#%% main function

def build_GSEP_extended(
    xray_flux: bool = False,
    ar_info: bool = False,
    sunspot_number: bool = False,
    flags: bool = False,
    slice_range: bool = False,
    ref1: bool = False,
    ref2: bool = False,
    numeric_conversion: bool = True,
) -> pd.DataFrame:
    """
    Build the GSEP_extended dataset, adding only the requested characteristics.

    Every argument is an optional boolean (default False) turning ON one
    feature, implemented by its own function above. Nothing is computed
    unless its flag is True, so build_GSEP_extended() with no arguments
    just returns the raw GSEP list untouched.

    Parameters
    ----------
    xray_flux : bool
        Calls add_xray_flux -> 'fl_goes_xray'.
    ar_info : bool
        Calls add_ar_info -> AR location/area/magnetic type/Z, Zurich
        lookup table, rank-encoded columns.
    sunspot_number : bool
        Calls add_sunspot_number -> 'daily_sn'.
    flags : bool
        Calls add_flags -> CME/flare/radio-burst flags, S1-S4 classes.
    slice_range : bool
        Calls add_slice_range -> 'slice range'.
    ref1 : bool
        Calls add_ref1 -> time differences (minutes) vs 'timestamp'.
    ref2 : bool
        Calls add_ref2 -> time differences (minutes) vs 'fl_start_time'.
    numeric_conversion : bool
        Calls convert_numeric_types on the modelling columns.

    Returns
    -------
    pd.DataFrame
        GSEP_extended, with the requested columns added.
    """

    GSEP_extended = dataset_reading.load_GSEP()

    if xray_flux:
        GSEP_extended = add_xray_flux(GSEP_extended)

    if ar_info:
        GSEP_extended = add_ar_info(GSEP_extended)

    if sunspot_number:
        GSEP_extended = add_sunspot_number(GSEP_extended)

    if flags:
        GSEP_extended = add_flags(GSEP_extended)

    if slice_range:
        GSEP_extended = add_slice_range(GSEP_extended)

    if ref1:
        GSEP_extended = add_ref1(GSEP_extended)

    if ref2:
        GSEP_extended = add_ref2(GSEP_extended)

    if numeric_conversion:
        GSEP_extended = convert_numeric_types(GSEP_extended)

    return GSEP_extended

#%% GSEP_int

def build_GSEP_int_extended(
    xray_flux: bool = False,
    ar_info: bool = False,
    sunspot_number: bool = False,
    flags: bool = False,
    slice_range: bool = False,
    ref1: bool = False,
    ref2: bool = False,
    numeric_conversion: bool = True,
) -> pd.DataFrame:
    GSEP_extended = build_GSEP_extended(
        xray_flux = False,
        ar_info = False,
        sunspot_number = False,
        flags = False,
        slice_range = False,
        ref1 = False,
        ref2 = False,
        numeric_conversion = True,
    )
    return GSEP_extended.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')

#%% ARs 