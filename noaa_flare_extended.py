import pandas as pd
import numpy as np
from datetime import date


from usefull_functions import time_mean
import dataset_reading

#%%

def merge_ar_info(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge AR (Active Region) info from df2 into df1 based on:
      1. Same calendar date between df1['time_start'] and df2['DATETIME']
      2. AR number matching between df2['AR_number'] and df1['nar']:
             noaa_ar == AR_number   OR   noaa_ar == AR_number + 10000
         (checked unconditionally, no date restriction)

    If exactly one row of df2 survives the filtering, its info is copied
    into new columns of df1:
        Location  -> AR_location
        Lo        -> AR_lo
        Area      -> AR_area
        Z         -> AR_z
        LL        -> AR_ll
        NN        -> AR_nn
        Mag_type  -> AR_Hale
    If zero rows match, the new columns are left as NaN for that row.
    If more than one row matches (ambiguous), a warning is printed and
    the first match is used.
    """
    df1 = df1.copy()
    df2 = df2.copy()

    # --- Ensure proper datetime dtypes ---
    df1['time_start'] = pd.to_datetime(df1['time_start'])
    df2['DATETIME'] = pd.to_datetime(df2['DATETIME'])
    # Precompute date-only column in df2 once, for fast filtering
    df2['_date_only'] = df2['DATETIME'].dt.date

    # New columns to fill in df1 (NaN by default = "no match")
    new_cols = ['AR_Location', 'AR_Lo', 'AR_Area', 'AR_Mcintosh',
                'AR_LL', 'AR_NN', 'AR_Hale']
    for col in new_cols:
        df1[col] = np.nan

    total = len(df1)
    for pos, (idx, row) in enumerate(df1.iterrows(), start=1):
        print(f"line {pos} over {total}")
        fl_date = row['time_start'].date()
        noaa_ar = row['nar']

        # --- Step 2: same date filter ---
        candidates = df2[df2['_date_only'] == fl_date]
        if candidates.empty:
            continue  # columns stay NaN

        # --- Step 3: AR_number == nar  OR  AR_number + 10000 == nar ---
        mask = (candidates['AR_number'] == noaa_ar) | \
               ((candidates['AR_number'] + 10000) == noaa_ar)

        matched = candidates[mask]
        if matched.empty:
            continue  # columns stay NaN

        if len(matched) > 1:
            print(f"  -> WARNING: {len(matched)} ambiguous matches for "
                  f"df1 row {idx} (date={fl_date}, noaa_ar={noaa_ar}). "
                  f"Using the first match.")
        match_row = matched.iloc[0]

        # --- Step 4: copy info over ---
        df1.at[idx, 'AR_location'] = match_row['Location'].astype(object)
        df1.at[idx, 'AR_lo'] = match_row['Lo'].astype(object)
        df1.at[idx, 'AR_area'] = match_row['Area'].astype(object)
        df1.at[idx, 'AR_z'] = match_row['Z'].astype(object)
        df1.at[idx, 'AR_ll'] = match_row['LL'].astype(object)
        df1.at[idx, 'AR_nn'] = match_row['NN'].astype(object)
        df1.at[idx, 'AR_Hale'] = match_row['Mag_type'].astype(object)

    return df1
#%%

noaa_flares = dataset_reading.load_noaa_flares()
srs_combine_complete = dataset_reading.load_srs_combine_complete()
srs_combine_complete['Mag_type'] = srs_combine_complete['Mag_type'].str.upper()
noaa_flares_extended = merge_ar_info(noaa_flares, srs_combine_complete)
noaa_flares_extended.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/noaa_flares_extended.pkl")