from __future__ import annotations

import sys
import pandas as pd
import numpy as np
import re

sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

from usefull_functions import time_mean, convert_prefix_value
import dataset_reading

#%%X ray flux

def add_xray_flux(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'fl_goes_xray' (numeric GOES X-ray class, from 'fl_goes_class')."""
    df = df.copy()
    df['fl_goes_xray'] = df['fl_goes_class'].apply(convert_prefix_value)       #function in usefull_functions.py, cell Flares Location
    return df

#%%AR

def merge_ar_info(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge AR (Active Region) info from df2 into df1 based on:
      1. Same calendar date between df1['fl_start_time'] and df2['DATETIME']
      2. df2['AR_number'] + k*10000 == df1['noaa_ar'], for k in (0, 1)

    If exactly one row of df2 survives the filtering, its info is copied
    into new columns of df1:
        Location  -> AR_Location
        Lo        -> AR_Lo
        Area      -> AR_Area
        Z         -> AR_Mcintosh
        LL        -> AR_LL
        NN        -> AR_NN
        Mag_type  -> AR_Hale

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
    new_cols = ['AR_Location', 'AR_Lo', 'AR_Area', 'AR_Mcintosh',
                'AR_LL', 'AR_NN', 'AR_Hale']
    for col in new_cols:
        df1[col] = None

    total = len(df1)

    # Use positional enumeration so the progress counter is always 1..total,
    # regardless of df1's actual index values.
    for pos, (idx, row) in enumerate(df1.iterrows(), start=1):

        fl_date = row['fl_start_time'].date()
        noaa_ar = row['noaa_ar']

        # --- Step 2: same date filter ---
        candidates = df2[df2['_date_only'] == fl_date]
        if candidates.empty:
            continue

        # --- Step 3: AR_number + k*10000 == noaa_ar, k = 0 or 1 ---
        mask = (candidates['AR_number_corrected'] == noaa_ar)
        matched = candidates[mask]

        if matched.empty:
            continue

        if len(matched) > 1:
            print(f"  -> WARNING: {len(matched)} ambiguous matches for "
                  f"df1 row {idx} (date={fl_date}, noaa_ar={noaa_ar}). "
                  f"Using the first match.")

        match_row = matched.iloc[0]

        # --- Step 4: copy info over ---
        df1.at[idx, 'AR_Location'] = match_row['Location']
        df1.at[idx, 'AR_Lo'] = match_row['Lo']
        df1.at[idx, 'AR_Area'] = match_row['Area']
        df1.at[idx, 'AR_Mcintosh'] = match_row['Z']
        df1.at[idx, 'AR_LL'] = match_row['LL']
        df1.at[idx, 'AR_NN'] = match_row['NN']
        df1.at[idx, 'AR_Hale'] = match_row['Mag_type']

    return df1

def add_ar_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge NOAA Active Region info, map it to integer codes, merge the
    Zurich classification lookup table, and rank-encode the resulting
    categorical columns against 'noaa_pf10MeV'.
    """
    df = df.copy()
    srs_combine_complete_corrected = dataset_reading.load_srs_combine_complete_corrected()

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

    df = merge_ar_info(df, srs_combine_complete_corrected)  # own function

    df['AR_Hale'] = df['AR_Hale'].str.upper()  # put str in CAPITAL
    df['AR_Hale_int'] = df['AR_Hale'].map(mapping)

    df['AR_Mcintosh'] = df['AR_Mcintosh'].str.upper()
    df['AR_Mcintosh_int'] = df['AR_Mcintosh'].map(mapping2)

    df['AR_Mcintosh_Z'] = df['AR_Mcintosh'].str[0]
    df['AR_Mcintosh_p'] = df['AR_Mcintosh'].str[1]
    df['AR_Mcintosh_c'] = df['AR_Mcintosh'].str[2]

    df['AR_Mcintosh_Z_int'] = df['AR_Mcintosh_Z'].map(mapping_letter1)
    df['AR_Mcintosh_p_int'] = df['AR_Mcintosh_p'].map(mapping_letter2)
    df['AR_Mcintosh_c_int'] = df['AR_Mcintosh_c'].map(mapping_letter3)

    # 1. Load the lookup table from the sheet provided
    lookup = pd.read_excel(
        "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/zurich classification parameters.xlsx",
        sheet_name="Sheet1",
        usecols=["ZMcI-type", "Magnetic type", "Length", "Penumbra type", "Distribution"]
    )

    # 2. Rename lookup columns to match desired output names,
    #    and rename the key column to match df's column name (AR_Mcintosh)
    lookup = lookup.rename(columns={
        "ZMcI-type": "AR_Mcintosh",
        "Magnetic type": "AR_Mcintosh_magnetic_type",
        "Length": "AR_Mcintosh_length",
        "Penumbra type": "AR_Mcintosh_penumbra_type",
        "Distribution": "AR_Mcintosh_distribution",
    })

    lookup["AR_Mcintosh"] = lookup["AR_Mcintosh"].str.upper()

    # 3. Merge on AR_Mcintosh (left join keeps all rows of df, including NaNs / unmatched codes)
    df = df.merge(lookup, on="AR_Mcintosh", how="left")

    cols_to_rank = [
        'AR_Mcintosh',
        'AR_Hale',
        'AR_Mcintosh_Z',
        'AR_Mcintosh_p',
        'AR_Mcintosh_c',
        'AR_Mcintosh_magnetic_type',
        'AR_Mcintosh_length',
        'AR_Mcintosh_penumbra_type',
        'AR_Mcintosh_distribution',
    ]

    for col in cols_to_rank:
        sorted_categories = df.groupby(col)['noaa_pf10MeV'].mean().sort_values().index
        new_mapping = {cat: rank for rank, cat in enumerate(sorted_categories, start=1)}
        df[f'{col}_int_ranked'] = df[col].map(new_mapping)
    
    #Convert location
    lat_pattern = r'([NS])(\d+\.?\d*)'
    long_pattern = r'([EW])(\d+\.?\d*)'
    
    def parse(s):
        if not isinstance(s, str):
            return pd.Series([None, None])
        
        # Latitude
        lat_match = re.search(lat_pattern, s)
        if lat_match:
            lat_sign, lat_val = lat_match.groups()
            lat = float(lat_val) * (1 if lat_sign == 'N' else -1)
        else:
            lat = None
        
        # Longitude
        long_match = re.search(long_pattern, s)
        if long_match:
            long_sign, long_val = long_match.groups()
            long = float(long_val) * (1 if long_sign == 'W' else -1)
        else:
            long = None
        
        return pd.Series([lat, long])
    
    df[['AR_lat', 'AR_long']] = df['AR_Location'].apply(parse)

    return df

#%%Sunspot Numbers (SN)

def merge_daily_sn(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sunspot number info from df2 into df1 based on:
      Same calendar date between df1['fl_start_time'] (or df1['timestamp'] fallback) 
      and df2['datetime']

    df2['datetime'] is built from its 'year', 'month', 'day' columns,
    in the format YYYY-MM-DD 00:00:01.
    """

    df1 = df1.copy()
    df2 = df2.copy()

    # --- Build the datetime column in df2 from year/month/day ---
    df2['datetime'] = pd.to_datetime(
        dict(year=df2['year'], month=df2['month'], day=df2['day'])
    ) + pd.Timedelta(seconds=1)  # -> YYYY-MM-DD 00:00:01

    # Ensure fl_start_time exists, fallback to timestamp if missing or NaT/NaN
    if 'fl_start_time' in df1.columns:
        df1['fl_start_time'] = df1['fl_start_time'].fillna(df1['timestamp'])
    else:
        df1['fl_start_time'] = df1['timestamp']

    # Convert fl_start_time to proper datetime dtype
    df1['fl_start_time'] = pd.to_datetime(df1['fl_start_time'])

    # Precompute date-only column in df2 once, for fast filtering
    df2['_date_only'] = df2['datetime'].dt.date

    # New column to fill in df1
    df1['daily_sn'] = None

    for idx, row in df1.iterrows():
        if pd.isna(row['fl_start_time']):
            continue

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

def add_sunspot_number(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the daily total sunspot number into 'daily_sn'."""
    df = df.copy()
    SN_d_tot_V2 = dataset_reading.load_SN_d_tot_V2()
    df = merge_daily_sn(df, SN_d_tot_V2)

    del SN_d_tot_V2
    return df

#%% Flares 

def match_flares_to_events(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    window_minutes: int = 30,
    flag_col: str = "flare flag",
    time_col_a: str = "fl_start_time",
    time_col_b: str = "time_start",
    ar_col_a: str = "noaa_ar",
    ar_col_b: str = "AR_number_corrected",
    xray_col_a: str = "fl_goes_xray",
    xray_col_b: str = "xray_flux",
    b_columns: list[str] | None = None,
    b_prefix: str = "noaa_flares_",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match each flagged event in df1 to at most one event in df2.
 
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Type A and type B event tables.
    window_minutes : int, default 30
        Half-width of the matching time window, in minutes. The window
        checked is [fl_start_time - window_minutes, fl_start_time +
        window_minutes], so 30 -> a 1-hour window total.
    flag_col, time_col_a, ar_col_a, xray_col_a : str
        Column names to read from df1.
    time_col_b, ar_col_b, xray_col_b : str
        Column names to read from df2.
    b_columns : list of str, optional
        Which df2 columns to copy into the enriched output. Defaults to
        None, which copies every column in df2. The columns used for
        matching (time_col_b, ar_col_b, xray_col_b) are always used
        internally regardless of this setting -- this only controls what
        gets attached to the final result.
    b_prefix : str, default "b_"
        Prefix applied to df2 columns when attached to the enriched
        output, to avoid name collisions with df1 columns.
 
    Returns
    -------
    enriched : pd.DataFrame
        Copy of df1 with matched df2 columns appended (prefixed with
        `b_prefix`) and a `match_index` column holding the matched df2
        row's original index label (NaN where there is no match).
    stage_counts : pd.DataFrame
        Same index and number of rows as df1. Columns:
            - n_time_window : # df2 rows inside the time window
            - n_ar_match    : # of those with a matching AR number
            - n_xray_match  : 1 if a final match was selected, else 0
        Rows where flag_col != 1 (or fl_start_time is missing) are all
        zero, since matching was skipped for them.
    """
    df1 = df1.copy()
    df2 = df2.copy()
 
    df1[time_col_a] = pd.to_datetime(df1[time_col_a])
    df2[time_col_b] = pd.to_datetime(df2[time_col_b])
 
    n = len(df1)
    n_time_window = np.zeros(n, dtype=int)
    n_ar_match = np.zeros(n, dtype=int)
    n_xray_match = np.zeros(n, dtype=int)
    match_index = pd.Series(np.nan, index=df1.index, dtype="object")
 
    for pos, (i, row) in enumerate(df1.iterrows()):
        if row[flag_col] != 1:
            continue
 
        t0 = row[time_col_a]
        if pd.isna(t0):
            continue
 
        window_start = t0 - pd.Timedelta(minutes=window_minutes)
        window_end = t0 + pd.Timedelta(minutes=window_minutes)
 
        # Stage 1: time window
        time_candidates = df2[
            (df2[time_col_b] >= window_start) & (df2[time_col_b] <= window_end)
        ]
        n_time_window[pos] = len(time_candidates)
        if time_candidates.empty:
            continue
 
        # Stage 2: AR number match
        ar_candidates = time_candidates[time_candidates[ar_col_b] == row[ar_col_a]]
        n_ar_match[pos] = len(ar_candidates)
        if ar_candidates.empty:
            continue
 
        # Stage 3: closest x-ray flux (only matters if >1 candidate)
        if len(ar_candidates) == 1:
            chosen = ar_candidates.index[0]
        else:
            diffs = (ar_candidates[xray_col_b] - row[xray_col_a]).abs()
            if diffs.notna().any():
                chosen = diffs.idxmin()
            else:
                # No usable x-ray values to compare; fall back to first candidate
                chosen = ar_candidates.index[0]
 
        n_xray_match[pos] = 1
        match_index.iloc[pos] = chosen
 
    stage_counts = pd.DataFrame(
        {
            "n_time_window": n_time_window,
            "n_ar_match": n_ar_match,
            "n_xray_match": n_xray_match,
        },
        index=df1.index,
    )
 
    df2_for_merge = df2[b_columns] if b_columns is not None else df2
    matched_b = df2_for_merge.reindex(match_index.values)
    matched_b.index = df1.index
    matched_b = matched_b.add_prefix(b_prefix)
 
    enriched = pd.concat([df1, matched_b], axis=1)
 
    return enriched, stage_counts

noaa_flares_c1 = dataset_reading.load_noaa_flares_c1()

#%%Slice Range 
  
def add_slice_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'slice range': duration in minutes between slice_start and slice_end."""
    df = df.copy()
    _, df_clean, _ = time_mean(df['slice_start'], df['slice_end'], diff_max=1000) #function in usefull_functions.py, cell Flares Events Time
    diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(df.index) 
    df['slice range'] = diff_minutes
    return df

#%%Flags (CME, flares, radioburst, S-storm level, SEP type)

def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add CME / flare / radio-burst flags and the SEP intensity classes."""
    df = df.copy()

    df['CME flag'] = df['cme_id'].notna().astype(int)

    df['flare flag'] = df['fl_start_time'].notna().astype(int)

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

def add_sep_type(df: pd.DataFrame) -> pd.DataFrame:
    """Threshold: 24h --> Papaioannou et al. (2025)"""
    df = df.copy()
    is_impulsive = df['slice range'] <= 24 * 60
    df['sep type str'] = np.where(is_impulsive, 'impulsive', 'gradual')
    df['sep type int'] = np.where(is_impulsive, 0, 1)
    return df

#%% Time differences (ref1, ref2)

def add_ref1(df: pd.DataFrame) -> pd.DataFrame:
    """Add time differences (minutes) of several reference times vs 'timestamp'."""
    df = df.copy()
    for col, out_name in [
        ('cdaw_start_time', 'cdaw_start_time ref1'),
        ('cme_1st_app_time', 'cme_1st_app_time ref1'),
        ('cme_launch_time', 'cme_launch_time ref1'),
        ('fl_start_time', 'fl_start_time ref1'),
        ('fl_peak_time', 'fl_peak_time ref1'),
    ]:
        _, df_clean, _ = time_mean(df[col], df['timestamp'], diff_max=1000)    #function in usefull_functions.py, cell Flares Events Time
        diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(df.index)
        df[out_name] = diff_minutes
    return df

def add_ref2(df: pd.DataFrame) -> pd.DataFrame:
    """Add time differences (minutes) of several reference times vs 'fl_start_time'."""
    df = df.copy()
    for col, out_name in [
        ('cdaw_start_time', 'cdaw_start_time ref2'),
        ('cme_1st_app_time', 'cme_1st_app_time ref2'),
        ('cme_launch_time', 'cme_launch_time ref2'),
        ('fl_peak_time', 'fl_peak_time ref2'),
        ('timestamp', 'timestamp ref2'),
    ]:
        _, df_clean, _ = time_mean(df[col], df['fl_start_time'], diff_max=1000) #function in usefull_functions.py, cell Flares Events Time
        diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(df.index)
        df[out_name] = diff_minutes
    return df

#%% NaN conversion for missing values

def convert_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force numeric dtype (pd.to_numeric, invalid -> NaN) on the columns used
    for modelling. Columns coming from a disabled feature (e.g. 'daily_sn'
    if add_sunspot_number was not applied) are silently skipped.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

#%% main functions (GSEP & GSEP_int)

def build_GSEP_extended(
    xray_flux: bool = True,
    ar_info: bool = True,
    sunspot_number: bool = True,
    flares_info: bool = True,
    flags: bool = True,
    slice_range: bool = True,
    sep_type: bool = True, 
    ref1: bool = True,
    ref2: bool = True,
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
        
    if flares_info:
        GSEP_extended, _ = match_flares_to_events(GSEP_extended, noaa_flares_c1, 
                                                  b_columns = ['long_carr', 'optical_class', 'flares_count_last24h', 
                                                               'xray_average_last24h', 'xray_max_last24h', 'xray_sum_last24h', 
                                                               'AR_flares_count_last24h', 'AR_xray_average_last24h', 
                                                               'AR_xray_max_last24h', 'AR_xray_sum_last24h', 
                                                               'flares_count_last48h', 
                                                                            'xray_average_last48h', 'xray_max_last48h', 'xray_sum_last48h', 
                                                                            'AR_flares_count_last48h', 'AR_xray_average_last48h', 
                                                                            'AR_xray_max_last48h', 'AR_xray_sum_last48h',  'hec_id', 'time_end'])
    GSEP_extended = GSEP_extended.rename(columns={'noaa_flares_time_end' : 'fl_end_time'})
    
    if slice_range:
        GSEP_extended = add_slice_range(GSEP_extended)
        
    if sep_type:
        GSEP_extended = add_sep_type(GSEP_extended)

    if ref1:
        GSEP_extended = add_ref1(GSEP_extended)

    if ref2:
        GSEP_extended = add_ref2(GSEP_extended)

    # if numeric_conversion:
    #     GSEP_extended = convert_numeric_types(GSEP_extended)

    return GSEP_extended

def build_GSEP_int_extended(
    xray_flux = True,
    ar_info = True,
    sunspot_number = True,
    flares_info = True, 
    flags = True,
    slice_range = True,
    sep_type = True,
    ref1 = True,
    ref2 = True,
    numeric_conversion = True,
):
    GSEP_extended = build_GSEP_extended(
        xray_flux = xray_flux,
        ar_info = ar_info,
        sunspot_number = sunspot_number,
        flares_info= flares_info, 
        flags = flags,
        slice_range = slice_range,
        ref1 = ref1,
        ref2 = ref2,
        numeric_conversion = numeric_conversion,
    )
    return GSEP_extended.select_dtypes(include=['int', 'float'])
