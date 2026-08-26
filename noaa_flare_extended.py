import pandas as pd
import numpy as np
# from datetime import date


# from usefull_functions import time_mean
import dataset_reading
from usefull_functions import convert_prefix_value

#%% Datasets reading

noaa_flares = dataset_reading.load_noaa_flares()
srs_combine_complete_corrected = dataset_reading.load_srs_combine_complete_corrected()
SN_d_tot_V2 = dataset_reading.load_SN_d_tot_V2()

#%% Continuous X ray

noaa_flares['xray_flux'] = noaa_flares['xray_class'].apply(convert_prefix_value) 

#%% ARs  /!\ Takes a while

# AR Number corrected
noaa_flares['time_start'] = pd.to_datetime(noaa_flares['time_start'])
condition = (noaa_flares['time_start'].dt.year >= 2002) & (noaa_flares['nar'] < 4000)
noaa_flares['AR_number_corrected'] = np.where(condition & noaa_flares['nar'].notna(), noaa_flares['nar'] + 10000, noaa_flares['nar'])

# AR srs informations 
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
    df1 = df1.reset_index(drop=True)
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
        print("processing the merge of AR info, can take a while")
        fl_date = row['time_start'].date()
        nar = row['AR_number_corrected']

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
        
        df1['AR_Mcintosh'] = df1['AR_Mcintosh'].str.upper()


    return df1
noaa_flares = merge_ar_info(noaa_flares, srs_combine_complete_corrected)

#%% Sunspot Numbers (SN)  /!\ Takes a while

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
        print("processing the merge of SN info, can take a while") 

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
noaa_flares = merge_daily_sn(noaa_flares, SN_d_tot_V2)

#%% Rolling count

def rolling_window_stats(df, time_col, value_col, group_col=None, window='24h', closed='left'):
    """
    Calcule count / mean / max / sum de value_col sur une fenêtre glissante temporelle
    [t-window, t) (grâce à closed='left'), globalement ou par groupe.
    Retourne un DataFrame aligné sur l'index de df, avec les colonnes
    ['count', 'mean', 'max', 'sum'].
    """
    def _stats(g):
        r = g.rolling(window, on=time_col, closed=closed)[value_col]
        return pd.DataFrame(
            {'count': r.count(), 'mean': r.mean(), 'max': r.max(), 'sum': r.sum()},
            index=g.index,  # préserve les vrais indices d'origine (uniques)
        )

    if group_col is None:
        return _stats(df)

    return (
        df.groupby(group_col, dropna=False, group_keys=False)
        .apply(_stats)
    )

# --- Préparation ---
noaa_flares['time_start'] = pd.to_datetime(noaa_flares['time_start'])
noaa_flares = noaa_flares.sort_values('time_start').reset_index(drop=True)

# --- Statistiques globales sur 24h ---
global_stats_24h = rolling_window_stats(noaa_flares, 'time_start', 'xray_flux', window='24h')
noaa_flares['flares_count_last24h'] = global_stats_24h['count']
noaa_flares['xray_average_last24h'] = global_stats_24h['mean']
noaa_flares['xray_max_last24h']     = global_stats_24h['max']
noaa_flares['xray_sum_last24h']     = global_stats_24h['sum']

# --- Statistiques globales sur 48h ---
global_stats_48h = rolling_window_stats(noaa_flares, 'time_start', 'xray_flux', window='48h')
noaa_flares['flares_count_last48h'] = global_stats_48h['count']
noaa_flares['xray_average_last48h'] = global_stats_48h['mean']
noaa_flares['xray_max_last48h']     = global_stats_48h['max']
noaa_flares['xray_sum_last48h']     = global_stats_48h['sum']

# --- Statistiques par région active (AR) sur 24h ---
ar_stats_24h = rolling_window_stats(
    noaa_flares, 'time_start', 'xray_flux', group_col='AR_number_corrected', window='24h'
)
noaa_flares['AR_flares_count_last24h'] = ar_stats_24h['count']
noaa_flares['AR_xray_average_last24h'] = ar_stats_24h['mean']
noaa_flares['AR_xray_max_last24h']     = ar_stats_24h['max']
noaa_flares['AR_xray_sum_last24h']     = ar_stats_24h['sum']

# --- Statistiques par région active (AR) sur 48h ---
ar_stats_48h = rolling_window_stats(
    noaa_flares, 'time_start', 'xray_flux', group_col='AR_number_corrected', window='48h'
)
noaa_flares['AR_flares_count_last48h'] = ar_stats_48h['count']
noaa_flares['AR_xray_average_last48h'] = ar_stats_48h['mean']
noaa_flares['AR_xray_max_last48h']     = ar_stats_48h['max']
noaa_flares['AR_xray_sum_last48h']     = ar_stats_48h['sum']

#%% Export
noaa_flares.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/noaa_flares_extended.pkl")

noaa_flares_c1_threshold = noaa_flares[noaa_flares['xray_flux'] >= 1e-6]   
noaa_flares_c1_threshold.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/noaa_flares_c1_threshold.pkl")

noaa_flares = dataset_reading.load_noaa_flares_extended()
