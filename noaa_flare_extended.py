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
      2. df2['AR_number'] + k*10000 == df1['nar'], for k in (0, 1)

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
    df1['time_start'] = pd.to_datetime(df1['time_start'])
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
        print(f"line {pos} over {total}")

        fl_date = row['time_start'].date()
        nar = row['nar']

        # --- Step 2: same date filter ---
        candidates = df2[df2['_date_only'] == fl_date]
        if candidates.empty:
            continue

        # --- Step 3: AR_number + k*10000 == nar, k = 0 or 1 ---
        mask = (candidates['AR_number_corrected'] == nar)
        matched = candidates[mask]

        if matched.empty:
            continue

        if len(matched) > 1:
            print(f"  -> WARNING: {len(matched)} ambiguous matches for "
                  f"df1 row {idx} (date={fl_date}, nar={nar}). "
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


def merge_daily_sn(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sunspot number info from df2 into df1 based on:
      Same calendar date between df1['time_start'] and df2['datetime']

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
    df1['time_start'] = pd.to_datetime(df1['time_start'])

    # Precompute date-only column in df2 once, for fast filtering
    df2['_date_only'] = df2['datetime'].dt.date

    # New column to fill in df1
    df1['daily_sn'] = None

    total = len(df1)

    for pos, (idx, row) in enumerate(df1.iterrows(), start=1):
        print(f"line {pos} over {total}")

        fl_date = row['time_start'].date()

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

#%%

noaa_flares = dataset_reading.load_noaa_flares()
srs_combine_complete_corrected = dataset_reading.load_srs_combine_complete_corrected()

noaa_flares_extended = merge_ar_info(noaa_flares, srs_combine_complete_corrected)
# noaa_flares_extended.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/noaa_flares_extended.pkl")