#%% sys

import sys
import pandas as pd

sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/')
from usefull_functions import time_mean, convert_prefix_value

sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/dataset access/')
import dataset_reading
import sep_dictionaries

#%% reading

GSEP_extended = dataset_reading.GSEP_list

#%% X ray flux

GSEP_extended['fl_goes_xray'] = GSEP_extended['fl_goes_class'].apply(convert_prefix_value)

#%% ARs

# srs_combine_complete = dataset_reading.srs_combine_complete


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

GSEP_extended = merge_ar_info(GSEP_extended, srs_combine_complete)


#%% Flags 

GSEP_extended['CME flag'] = GSEP_extended['cme_id'].notna().astype(int)

GSEP_extended['flare flag'] = GSEP_extended['fl_id'].notna().astype(int)

GSEP_extended['CME + flare flag'] = (GSEP_extended['cme_id'].notna() & GSEP_extended['fl_id'].notna()).astype(int)


GSEP_extended['radio burst 1'] = GSEP_extended['gsep_notes'].str.startswith('Type II Radio burst ').astype(int) #(from \'gsep_notes\')
GSEP_extended['radio burst 2'] = GSEP_extended['m_type2_onset_time'].notna().astype(int) # (from \'m_type2_onset_time\')

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