import pandas as pd
import re
from itertools import chain

import dataset_reading 

noaa_flares = dataset_reading.load_noaa_flares_extended()
GSEP = dataset_reading.load_GSEP_extended()
PRISM_1h = dataset_reading.load_PRISM_analyzed_rolling_combinded_seq_1hours()
PRISM_24h = dataset_reading.load_PRISM_analyzed_rolling_combinded_seq_24hours()

date_a = pd.Timestamp('1986-02-04') #GSEP & PRISM beginning
date_b = pd.Timestamp('2017-09-10') #GSEP end

noaa_flares = noaa_flares[noaa_flares['time_start'].between(date_a, date_b)]


import matplotlib.pyplot as plt
plt.hist(GSEP['cme_1st_app_time ref1'], bins = 100)

#%% 

GSEP_col = GSEP.columns.tolist()
cme_col = ['cme_1st_app_time', 'lasco_linear_speed', 'lasco_cme_width']
flare_col = ['noaa_flares_hec_id', 'fl_start_time', 'fl_peak_time', 'fl_lon', 'fl_lat', 'fl_goes_xray', 
             'noaa_flares_flares_count_last24h', 'noaa_flares_xray_average_last24h', 'noaa_flares_xray_max_last24h', 
             'noaa_flares_AR_flares_count_last24h', 'noaa_flares_AR_xray_average_last24h', 'noaa_flares_AR_xray_max_last24h']
AR_col = ['noaa_ar', 'AR_long', 'AR_lat', 'AR_Area', 'AR_Mcintosh' , 'AR_Hale']
SN_col = ['daily_sn']
proton_flux_col = ['noaa_pf10MeV']
flags_col = ['>= S1', '>= S2', '>= S3', '= S1', '= S2', '= S3', '= S4', 'S_class']

#%% GSEP "active days"
sep_pret_active = pd.DataFrame()

sep_pret_active = GSEP[list(chain.from_iterable([cme_col, flare_col, AR_col, SN_col, proton_flux_col, flags_col]))]


sep_pret_active['GSEP flag'] = 1
new_order = ["GSEP flag"] + [col for col in sep_pret_active.columns if col != "GSEP flag"]
sep_pret_active = sep_pret_active[new_order]

#%% noaa flares "quiet days"

noaa_flares_quiet = noaa_flares[~noaa_flares['hec_id'].isin(sep_pret_active['noaa_flares_hec_id'])]

sep_pret_quiet =  noaa_flares_quiet[[
    'hec_id', 'time_start', 'time_peak', 'AR_number_corrected', 'lat_hg', 
    'long_hg', 'xray_flux', 'flares_count_last24h', 'xray_average_last24h',
    'xray_max_last24h', 'AR_flares_count_last24h', 'AR_xray_average_last24h',
    'AR_xray_max_last24h', 'daily_sn', 'AR_Area', 'AR_Mcintosh', 'AR_Hale'
]].rename(columns={
    'hec_id' : 'noaa_flares_hec_id',
    'time_start': 'fl_start_time', 
    'time_peak': 'fl_peak_time',
    'AR_number_corrected': 'noaa_ar',
    'lat_hg': 'fl_lat',
    'long_hg': 'fl_lon',
    'xray_flux': 'fl_goes_xray',
    'flares_count_last24h': 'noaa_flares_flares_count_last24h',
    'xray_average_last24h': 'noaa_flares_xray_average_last24h',
    'xray_max_last24h': 'noaa_flares_xray_max_last24h',
    'AR_flares_count_last24h': 'noaa_flares_AR_flares_count_last24h',
    'AR_xray_average_last24h': 'noaa_flares_AR_xray_average_last24h',
    'AR_xray_max_last24h': 'noaa_flares_AR_xray_max_last24h',
    'daily_sn': 'daily_sn',
    'AR_Area': 'AR_Area',
    'AR_Mcintosh': 'AR_Mcintosh',
    'AR_Hale': 'AR_Hale'
})
    
sep_pret_quiet['>= S1'] = 0; sep_pret_quiet['>= S2'] = 0; sep_pret_quiet['>= S3'] = 0
sep_pret_quiet['= S1'] = 0; sep_pret_quiet['= S2'] = 0; sep_pret_quiet['= S3'] = 0; sep_pret_quiet['= S4'] = 0
sep_pret_quiet['S_class'] = 0

sep_pret_quiet['GSEP flag'] = 0
sep_pret_quiet['noaa_pf10MeV'] = 0

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
    new_cols = ['AR_Location',  'AR_Area', 'AR_Mcintosh', 'AR_Hale']
    for col in new_cols:
        df1[col] = None

    # total = len(df1)

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
        df1.at[idx, 'AR_Area'] = match_row['Area']
        df1.at[idx, 'AR_Mcintosh'] = match_row['Z']
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
    
    df = merge_ar_info(df, srs_combine_complete_corrected)  # own function
    df['AR_Hale'] = df['AR_Hale'].str.upper()  
    df['AR_Mcintosh'] = df['AR_Mcintosh'].str.upper()
        
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

# sep_pret_quiet = add_ar_info(sep_pret_quiet)

cdaw_cme = dataset_reading.load_cdaw_cme()
def merge_cme_info(sep_pret_quiet: pd.DataFrame, cdaw_cme: pd.DataFrame) -> pd.DataFrame:
    
    
    sep_pret_quiet = sep_pret_quiet.copy()
    cdaw_cme = cdaw_cme.copy()
 
    # --- Ensure proper datetime dtypes ---
    sep_pret_quiet['fl_start_time'] = pd.to_datetime(sep_pret_quiet['fl_start_time'])
    cdaw_cme['t_start'] = pd.to_datetime(cdaw_cme['t_start'])
 
    # On écarte les lignes de cdaw_cme sans cme_1st_app_time exploitable
    cdaw_cme = cdaw_cme.dropna(subset=['t_start'])
 
    # New columns to fill in sep_pret_quiet
    new_cols = ['t_start', 'v_lin', 'width']
    for col in new_cols:
        sep_pret_quiet[col] = None
 
    # Use positional enumeration so the progress counter is always 1..total,
    # regardless of sep_pret_quiet's actual index values.
    for pos, (idx, row) in enumerate(sep_pret_quiet.iterrows(), start=1):
        flare_time = row['fl_start_time']
 
        # --- Step 1: skip if no flare_start_time ---
        if pd.isna(flare_time):
            continue
 
        # --- Step 2: window filter -> évènements dans les 2h après le flare ---
        window_end = flare_time + pd.Timedelta(hours=2)
        candidates = cdaw_cme[(cdaw_cme['t_start'] >= flare_time) &
                          (cdaw_cme['t_start'] <= window_end)]
 
        if candidates.empty:
            continue
 
        # --- Step 3: si plusieurs évènements, on garde le plus proche ---
        time_diffs = (candidates['t_start'] - flare_time).abs()
        closest_pos = time_diffs.idxmin()
        match_row = candidates.loc[closest_pos]
 
        if len(candidates) > 1:
            print(f"  -> INFO: {len(candidates)} évènements CME trouvés pour "
                  f"sep_pret_quiet ligne {idx} (flare_start_time={flare_time}). "
                  f"On garde le plus proche ({match_row['cme_1st_app_time']}).")
 
        # --- Step 4: copy info over ---
        sep_pret_quiet.at[idx, 't_start'] = match_row['t_start']
        sep_pret_quiet.at[idx, 'v_lin'] = match_row['v_lin']
        sep_pret_quiet.at[idx, 'width'] = match_row['width']
 
    return sep_pret_quiet

sep_pret_quiet = merge_cme_info(sep_pret_quiet, cdaw_cme)


#%% Concatenation

sep_pret = pd.concat([sep_pret_active, sep_pret_quiet], join='outer', ignore_index=True)


