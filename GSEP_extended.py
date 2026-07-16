#%% sys

import sys
import pandas as pd
import numpy as np


sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

from usefull_functions import time_mean, convert_prefix_value
import dataset_reading

#%% functions 

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

#%% reading

GSEP_extended = dataset_reading.GSEP_list

#%% X ray flux

GSEP_extended['fl_goes_xray'] = GSEP_extended['fl_goes_class'].apply(convert_prefix_value)


#%% ARs

srs_combine_complete = dataset_reading.srs_combine_complete




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
    "AXX": 1,
    "BXO": 2,
    "BXI": 3,
    "HRX": 4,
    "CRO": 5,
    "CRI": 6,
    "HAX": 7,
    "CAO": 8,
    "CAI": 9,
    "HSX": 10,
    "CSO": 11,
    "CSI": 12,
    "DRO": 13,
    "ERO": 14,
    "FRO": 15,
    "DRI": 16,
    "ERI": 17,
    "FRI": 18,
    "DAO": 19,
    "EAO": 20,
    "FAO": 21,
    "DAI": 22,
    "EAI": 23,
    "FAI": 24,
    "DSO": 25,
    "ESO": 26,
    "FSO": 27,
    "DSI": 28,
    "ESI": 29,
    "FSI": 30,
    "DAC": 31,
    "EAC": 32,
    "FAC": 33,
    "DSC": 34,
    "ESC": 35,
    "FSC": 36,
    "HKX": 37,
    "CKO": 38,
    "CKI": 39,
    "HHX": 40,
    "CHO": 41,
    "CHI": 42,
    "DKO": 43,
    "EKO": 44,
    "FKO": 45,
    "DKI": 46,
    "EKI": 47,
    "FKI": 48,
    "DHO": 49,
    "EHO": 50,
    "FHO": 51,
    "DHI": 52,
    "EHI": 53,
    "FHI": 54,
    "DKC": 55,
    "EKC": 56,
    "FKC": 57,
    "DHC": 58,
    "EHC": 59,
    "FHC": 60
}

mapping_letter1 = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "H": 7
    }

mapping_letter2 = {
    "X": 1,
    "R": 2,
    "S": 3,
    "A": 4,
    "H": 5,
    "K": 6
    }

mapping_letter3 = {
    "X": 1,
    "O": 2,
    "I": 3,
    "C": 4
    }


GSEP_extended = merge_ar_info(GSEP_extended, srs_combine_complete)

GSEP_extended['AR_mag_type'] = GSEP_extended['AR_mag_type'].str.upper()
GSEP_extended['AR_mag_type_int'] = GSEP_extended['AR_mag_type'].map(mapping)

GSEP_extended['AR_z'] = GSEP_extended['AR_z'].str.upper()
GSEP_extended['AR_z_int'] = GSEP_extended['AR_z'].map(mapping2)


GSEP_extended['group_configuration'] = GSEP_extended['AR_z'].str[0]
GSEP_extended['largest_spot_type'] = GSEP_extended['AR_z'].str[1]
GSEP_extended['spots_distribution'] = GSEP_extended['AR_z'].str[2]


GSEP_extended['group_configuration_int'] = GSEP_extended['group_configuration'].map(mapping_letter1)
GSEP_extended['largest_spot_type_int'] = GSEP_extended['largest_spot_type'].map(mapping_letter2)
GSEP_extended['spots_distribution_int'] = GSEP_extended['spots_distribution'].map(mapping_letter3)



########################################################################################################
# 1. Load the lookup table from the sheet you provided
lookup = pd.read_excel(
    "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/zurich classification parameters.xlsx",
    sheet_name="Sheet1",
    usecols=["ZMcI-type", "Magnetic type", "Length", "Penumbra type", "Distribution"]
)

# 2. Rename lookup columns to match your desired output names,
#    and rename the key column to match your df's column name (AR_z)
lookup = lookup.rename(columns={
    "ZMcI-type": "AR_z",
    "Magnetic type": "AR_z_magnetic_type",
    "Length": "AR_z_length",
    "Penumbra type": "AR_z_penumbra_type",
    "Distribution": "AR_z_distribution",
})

lookup["AR_z"] = lookup["AR_z"].str.upper()

# 3. Merge on AR_z (left join keeps all rows of df, including NaNs / unmatched codes)
GSEP_extended = GSEP_extended.merge(lookup, on="AR_z", how="left")


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
    sorted_categories = GSEP_extended.groupby(col)['noaa_pf10MeV'].mean().sort_values().index
    new_mapping = {cat: rank for rank, cat in enumerate(sorted_categories, start=1)}
    GSEP_extended[f'{col}_int_ranked'] = GSEP_extended[col].map(new_mapping)

#%% sunspot numbers

SN_d_tot_V2 = pd.read_csv("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/sunspot/SN_d_tot_V2.0.csv", 
                           delimiter=";", 
                           header=None, 
                           names = ['year', 'month', 'day', 'date_decimal', 'daily_total_sn', 'daily_std_variation', 'nb_observations', 'definitive/provisional'])



GSEP_extended = merge_daily_sn(GSEP_extended, SN_d_tot_V2)



del SN_d_tot_V2
#%% Flags 

GSEP_extended['CME flag'] = GSEP_extended['cme_id'].notna().astype(int)

GSEP_extended['flare flag'] = GSEP_extended['fl_id'].notna().astype(int)

GSEP_extended['CME + flare flag'] = (GSEP_extended['cme_id'].notna() & GSEP_extended['fl_id'].notna()).astype(int)


GSEP_extended['radio burst 1'] = GSEP_extended['gsep_notes'].str.startswith('Type II Radio burst ').astype(int) #(from \'gsep_notes\')
GSEP_extended['radio burst 2'] = GSEP_extended['m_type2_onset_time'].notna().astype(int) # (from \'m_type2_onset_time\')

GSEP_extended['>= S1'] = (GSEP_extended['noaa_pf10MeV'] > 10).astype(int)
GSEP_extended['>= S2'] = (GSEP_extended['noaa_pf10MeV'] > 100).astype(int) 
GSEP_extended['>= S3'] = (GSEP_extended['noaa_pf10MeV'] > 1000).astype(int)

GSEP_extended['= S1'] = ((GSEP_extended['noaa_pf10MeV'] >= 10) & (GSEP_extended['noaa_pf10MeV'] <100)).astype(int)
GSEP_extended['= S2'] = ((GSEP_extended['noaa_pf10MeV'] >= 100) & (GSEP_extended['noaa_pf10MeV'] <1000)).astype(int)
GSEP_extended['= S3'] = ((GSEP_extended['noaa_pf10MeV'] >= 1000) & (GSEP_extended['noaa_pf10MeV'] <10000)).astype(int)
GSEP_extended['= S4'] = (GSEP_extended['noaa_pf10MeV'] > 10000).astype(int)

conditions = [
    (GSEP_extended['noaa_pf10MeV'] >= 10)    & (GSEP_extended['noaa_pf10MeV'] < 100),
    (GSEP_extended['noaa_pf10MeV'] >= 100)   & (GSEP_extended['noaa_pf10MeV'] < 1000),
    (GSEP_extended['noaa_pf10MeV'] >= 1000)  & (GSEP_extended['noaa_pf10MeV'] < 10000),
    (GSEP_extended['noaa_pf10MeV'] > 10000),
]
valeurs = [1, 2, 3, 4]

GSEP_extended['S_class'] = np.select(conditions, valeurs, default=0)

del conditions, valeurs, mapping, mapping2, mapping_letter1, mapping_letter2, mapping_letter3, new_mapping, sorted_categories, lookup, col, cols_to_rank

#%% slice range

_, df_clean, _ = time_mean(GSEP_extended['slice_start'], GSEP_extended['slice_end'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended['slice range'] = diff_minutes

#%% ref 1 = timestamp

_, df_clean, _ = time_mean(GSEP_extended ['cdaw_start_time'], GSEP_extended ['timestamp'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended .index)
GSEP_extended ['cdaw_start_time ref1'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended ['cme_launch_time'], GSEP_extended ['timestamp'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended ['cme_launch_time ref1'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended['fl_start_time'], GSEP_extended['timestamp'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended['fl_start_time ref1'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended['fl_peak_time'], GSEP_extended['timestamp'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended['fl_peak_time ref1'] = diff_minutes

#%% ref 2 = flare_start

_, df_clean, _ = time_mean(GSEP_extended ['cdaw_start_time'], GSEP_extended ['fl_start_time'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended .index)
GSEP_extended ['cdaw_start_time ref2'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended ['cme_launch_time'], GSEP_extended ['fl_start_time'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended ['cme_launch_time ref2'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended['fl_peak_time'], GSEP_extended['fl_start_time'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended['fl_peak_time ref2'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended['timestamp'], GSEP_extended['fl_start_time'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended['timestamp ref2'] = diff_minutes

del df_clean, diff_minutes


#%% numeric values 

GSEP_extended['p_cme_width'] = pd.to_numeric(GSEP_extended['p_cme_width'], errors = 'coerce')
GSEP_extended['p_cme_speed'] = pd.to_numeric(GSEP_extended['p_cme_width'], errors = 'coerce')
GSEP_extended['Inst_category'] = pd.to_numeric(GSEP_extended['Inst_category'], errors = 'coerce')
GSEP_extended["noaa_pf10MeV"] = pd.to_numeric(GSEP_extended["noaa_pf10MeV"], errors = 'coerce')
GSEP_extended["noaa-sep_flag"] = pd.to_numeric(GSEP_extended["noaa-sep_flag"], errors = 'coerce')
GSEP_extended['radio burst 1'] = pd.to_numeric(GSEP_extended['radio burst 1'], errors = 'coerce')
GSEP_extended['radio burst 2'] = pd.to_numeric(GSEP_extended['radio burst 2'], errors = 'coerce')
GSEP_extended['AR_mag_type_int'] = pd.to_numeric(GSEP_extended['AR_mag_type_int'], errors = 'coerce')
GSEP_extended['AR_area'] = pd.to_numeric(GSEP_extended['AR_area'], errors = 'coerce')
GSEP_extended['daily_sn'] = pd.to_numeric(GSEP_extended['daily_sn'], errors = 'coerce')


#%% GSEP_int

GSEP_int = GSEP_extended.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')

GSEP_int_filtered = GSEP_int.drop(columns=['>= S1', '>= S2', '>= S3', '= S1', 
                                           '= S2', '= S3', '= S4', 'S_class',
                                           'noaa_pf10MeV', 'ppf_gt10MeV', 'gsep_fluence_gt10MeV',
                                           'fluence_gt10MeV', 'fluence_gt30MeV', 'cdaw_evn_max', 
                                           'ppf_gt30MeV', 'fluence_gt100MeV', 'ppf_gt60MeV', 
                                           'fluence_gt60MeV', 'noaa-sep_flag', 'ppf_gt100MeV', 
                                           'gsep_pf_gt10MeV', 'Flag'])

AR_params = GSEP_extended[['AR_location', 'AR_lo', 'AR_area', 'AR_z', 'AR_ll', 'AR_nn',  'AR_mag_type', 'AR_mag_type_int', 'AR_mag_type_int_ranked', 
                  'AR_z_int', 'group_configuration', 'largest_spot_type', 'spots_distribution', 'group_configuration_int', 
                  'largest_spot_type_int', 'spots_distribution_int', 'AR_z_magnetic_type', 'AR_z_length', 
                  'AR_z_penumbra_type', 'AR_z_distribution', 'AR_z_int_ranked', 'group_configuration_int_ranked', 
                  'largest_spot_type_int_ranked', 'spots_distribution_int_ranked', 'AR_z_magnetic_type_int_ranked', 
                  'AR_z_length_int_ranked', 'AR_z_penumbra_type_int_ranked', 'AR_z_distribution_int_ranked']]

AR_params_int = AR_params.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')