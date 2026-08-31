#%% librairies 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from itertools import chain
from imblearn.over_sampling import RandomOverSampler

#%% datasets

import dataset_reading                                    # personal file to load datasets

noaa_flares = dataset_reading.load_noaa_flares_extended() # personal extended version
cdaw_cme = dataset_reading.load_cdaw_cme()                # original version
GSEP = dataset_reading.load_GSEP_extended()               # personal extended version
srs_combine = dataset_reading.load_srs_combine_complete_corrected() # personal extended version

# crop datasets on the same timerange 
date_a = pd.Timestamp('1996-01-01') #SRS_combine and CDAW beginning
date_b = pd.Timestamp('2017-09-10') #GSEP end

noaa_flares['time_start'] = pd.to_datetime(noaa_flares['time_start']); noaa_flares = noaa_flares[noaa_flares['time_start'].between(date_a, date_b)]
GSEP['timestamp'] = pd.to_datetime(GSEP['timestamp']); GSEP = GSEP[GSEP['timestamp'].between(date_a, date_b)]
cdaw_cme['t_start'] = pd.to_datetime(cdaw_cme['t_start']);cdaw_cme = cdaw_cme[cdaw_cme['t_start'].between(date_a, date_b)]
srs_combine['DATETIME'] = pd.to_datetime(srs_combine['DATETIME']);srs_combine = srs_combine[srs_combine['DATETIME'].between(date_a, date_b)]

del date_a, date_b 
#%% functions

def merge_ar_info(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:

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
                  f"On garde le plus proche ({match_row['t_start']}).")
 
        # --- Step 4: copy info over ---
        sep_pret_quiet.at[idx, 't_start'] = match_row['t_start']
        sep_pret_quiet.at[idx, 'v_lin'] = match_row['v_lin']
        sep_pret_quiet.at[idx, 'width'] = match_row['width']
 
    return sep_pret_quiet

def plot_repartition(df, bins=100, figsize=(8, 5), top_n=100,
                     gsep_col=">= S1", gsep_col_2=None, column_params=None):
    """
    Affiche la répartition (distribution) de chaque colonne d'un dataframe.

    - Colonnes "flag" (peu de valeurs discrètes) -> diagramme en barres
    - Colonnes numériques continues -> histogramme
    - Colonnes non numériques -> diagramme en barres des top_n catégories

    Si `gsep_col` est présente :
    - gsep_col == 0 -> bleu (arrière-plan)
    - gsep_col == 1 :
        - sans gsep_col_2 ou gsep_col_2 absent -> rouge (premier plan)
        - si gsep_col_2 est spécifiée :
            - gsep_col_2 == 0 -> rouge (milieu)
            - gsep_col_2 == 1 -> vert (premier plan)
    """
    def _get_bin_edges(data, bins, xscale):
        if hasattr(bins, "__len__"):
            return bins
        
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return bins
        
        if xscale == "log":
            positive = data[data > 0]
            if len(positive) == 0:
                return bins
            return np.geomspace(positive.min(), positive.max(), int(bins) + 1)
        
        return np.linspace(data.min(), data.max(), int(bins) + 1)

    def _sort_counts_increasing(counts):
        try:
            numeric_idx = pd.to_numeric(counts.index)
            order = np.argsort(numeric_idx.to_numpy())
            return counts.iloc[order]
        except (ValueError, TypeError):
            return counts

    column_params = column_params or {}
    has_split_1 = gsep_col in df.columns
    has_split_2 = has_split_1 and (gsep_col_2 is not None) and (gsep_col_2 in df.columns)

    if has_split_1:
        df0 = df[df[gsep_col] == 0]
        if has_split_2:
            # Séparation uniquement lorsque gsep_col == 1
            df1_sub0 = df[(df[gsep_col] == 1) & (df[gsep_col_2] == 0)]
            df1_sub1 = df[(df[gsep_col] == 1) & (df[gsep_col_2] == 1)]
        else:
            df1 = df[df[gsep_col] == 1]

    # Colonnes à exclure des séparations
    ignored_cols = {gsep_col}
    if gsep_col_2:
        ignored_cols.add(gsep_col_2)

    for col in df.columns:
        params = column_params.get(col, {})
        xscale = params.get("xscale")
        yscale = params.get("yscale")
        col_bins = params.get("bins", bins)
        normalize = params.get("normalize", False)

        plt.figure(figsize=figsize)
        series = df[col]
        unique_vals = series.dropna().unique()
        is_flag = pd.api.types.is_numeric_dtype(series) and set(unique_vals).issubset({0, 1, 2, 3, 4, 5, 6})

        # Configuration des sous-ensembles (données, couleur, alpha, label, zorder)
        if has_split_1 and col not in ignored_cols:
            if has_split_2:
                subsets = [
                    (df0[col], "tab:blue", 0.4, "<= S0", 2),
                    (df1_sub0[col], "tab:red", 0.65, "S1 & S2", 3),
                    (df1_sub1[col], "tab:green", 0.85, "S3 & S4", 4),
                ]
            else:
                subsets = [
                    (df0[col], "tab:blue", 0.5, f"'{gsep_col}' == 0", 2),
                    (df1[col], "tab:red", 0.75, f"'{gsep_col}' == 1", 3),
                ]
        else:
            subsets = [(series, "tab:blue", 0.85, None, 2)]

        plotted_labels = []

        if is_flag:
            all_vals = sorted(set(series.dropna().unique()))
            # Adaptabilité de la largeur selon le nombre de séries
            if len(subsets) == 3:
                widths = [0.60, 0.40, 0.20]
            elif len(subsets) == 2:
                widths = [0.55, 0.28]
            else:
                widths = [0.55]

            for (data, color, alpha, label, zorder), width in zip(subsets, widths):
                clean = data.dropna()
                counts = clean.value_counts().reindex(all_vals, fill_value=0)
                values = counts.values.astype(float)
                if normalize and len(clean) > 0:
                    values = values / len(clean) * 100
                plt.bar(counts.index.astype(str), values, width=width, color=color,
                        alpha=alpha, label=label, zorder=zorder)
                if label:
                    plotted_labels.append(label)

            plt.ylabel("Percentage of time (%)" if normalize else "Count")
            plt.yscale(yscale if yscale else "log")

        elif pd.api.types.is_numeric_dtype(series):
            edges = _get_bin_edges(series.dropna(), col_bins, xscale)

            for data, color, alpha, label, zorder in subsets:
                clean = data.dropna()
                if len(clean) == 0:
                    continue
                weights = np.full(len(clean), 100.0 / len(clean)) if normalize else None
                plt.hist(clean, bins=edges, weights=weights, color=color, alpha=alpha,
                         label=label, zorder=zorder)
                if label:
                    plotted_labels.append(label)

            plt.ylabel("Percentage of time (%)" if normalize else "Count")
            if xscale:
                plt.xscale(xscale)
            if yscale:
                plt.yscale(yscale)

        else:
            top_categories = series.value_counts(dropna=True).head(top_n).index
            if len(subsets) == 3:
                widths = [0.85, 0.55, 0.25]
            elif len(subsets) == 2:
                widths = [0.80, 0.42]
            else:
                widths = [0.80]

            for (data, color, alpha, label, zorder), width in zip(subsets, widths):
                clean = data.dropna()
                counts = clean.value_counts().reindex(top_categories, fill_value=0)
                counts = _sort_counts_increasing(counts)
                values = counts.values.astype(float)
                if normalize and len(clean) > 0:
                    values = values / len(clean) * 100
                plt.bar(counts.index.astype(str), values, width=width, color=color,
                        alpha=alpha, label=label, zorder=zorder)
                if label:
                    plotted_labels.append(label)

            plt.ylabel("Percentage of time (%)" if normalize else "Count")
            plt.xticks(rotation=45, ha="right")
            if yscale:
                plt.yscale(yscale)

        if plotted_labels:
            plt.legend()

        plt.title(col)
        plt.xlabel(col)
        plt.tight_layout()
        plt.grid()
        plt.rcParams['font.size'] = 18
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.show()


#%% ACTIVE DAYS (GSEP)

sep_pret_active = GSEP

# positive event flag (not always >= S1, see GSEP SEP definition)
sep_pret_active['GSEP flag'] = 1

#%% QUIET DAYS (NOAA FLARES)

#remove the flares that have led to SEP, using the hec_id parameter (has been added to GSEP with my extension)   
noaa_flares_quiet = noaa_flares[~noaa_flares['hec_id'].isin(sep_pret_active['noaa_flares_hec_id'])]

#rename parameters to respect the active days names (ensuring good concatenation)
sep_pret_quiet =  noaa_flares_quiet.rename(columns={
    'hec_id' : 'noaa_flares_hec_id',
    'time_start': 'fl_start_time', 
    'time_peak': 'fl_peak_time',
    'time_end': 'fl_end_time', 
    'AR_number_corrected': 'noaa_ar',   
    'lat_hg': 'fl_lat',
    'long_hg': 'fl_lon',
    'daily_sn': 'daily_sn',
    'AR_Area': 'AR_Area',
    'AR_Mcintosh': 'AR_Mcintosh',
    'AR_Hale': 'AR_Hale', 
    'xray_flux' : 'fl_goes_xray'
})

#putting the flags to 0 
sep_pret_quiet['>= S1'] = 0; sep_pret_quiet['>= S2'] = 0; sep_pret_quiet['>= S3'] = 0
sep_pret_quiet['= S1'] = 0; sep_pret_quiet['= S2'] = 0; sep_pret_quiet['= S3'] = 0; sep_pret_quiet['= S4'] = 0
sep_pret_quiet['S_class'] = 0; sep_pret_quiet['GSEP flag'] = 0; #/!\ I permit to put the same flag as the S0 storm from GSEP, estimating that we are not focus on this events
sep_pret_quiet['noaa_pf10MeV'] = 0 #assumption without background noise

#adding the AR information (Mcintosh, Hale, Location, Lat, Lon, Area)
sep_pret_quiet = add_ar_info(sep_pret_quiet) #/!\ takes a while

#adding the CME informations (t_start, v_lin, width)
sep_pret_quiet = merge_cme_info(sep_pret_quiet, cdaw_cme) #/!\ takes a while

#we rename them with GSEP convention names to match the concatenation
sep_pret_quiet =  sep_pret_quiet.rename(columns={
    't_start' : 'cme_1st_app_time',    #as defined by GSEP 
    'v_lin': 'lasco_linear_speed',     #as defined by GSEP 
    'width': 'lasco_cme_width',        #as defined by GSEP 
    'xray_sum_last24h': 'noaa_flares_xray_sum_last24h', 
    'flares_count_last48h': 'noaa_flares_flares_count_last48h',
    'xray_average_last48h': 'noaa_flares_xray_average_last48h',
    'xray_max_last48h': 'noaa_flares_xray_max_last48h',
    'xray_sum_last48h': 'noaa_flares_xray_sum_last48h',
    'noaa_flares_AR_xray_max_last24h': 'noaa_flares_AR_xray_max_last24h',
    'AR_xray_sum_last24h': 'noaa_flares_AR_xray_sum_last24h',
    'AR_flares_count_last48h': 'noaa_flares_AR_flares_count_last48h',
    'AR_xray_average_last48h': 'noaa_flares_AR_xray_average_last48h',
    'AR_xray_max_last48h': 'noaa_flares_AR_xray_max_last48h',
    'AR_xray_sum_last48h': 'noaa_flares_AR_xray_sum_last48h', 
    'flares_count_last24h': 'noaa_flares_flares_count_last24h',
    'xray_average_last24h':'noaa_flares_xray_average_last24h', 
    'xray_max_last24h' : 'noaa_flares_xray_max_last24h', 
    'AR_flares_count_last24h' : 'noaa_flares_AR_flares_count_last24h', 
    'AR_xray_average_last24h': 'noaa_flares_AR_xray_average_last24h', 
    'AR_xray_max_last24h' : 'noaa_flares_AR_xray_max_last24h'
})

del noaa_flares_quiet
#%% SEP PRET (complete) 

#concatenation of active and quiet days, removing non-common columns
sep_pret = pd.concat([sep_pret_active, sep_pret_quiet], join='inner', ignore_index=True) #removing non-common columns


# type conversion
##datetime
sep_pret["fl_start_time"] = pd.to_datetime(sep_pret["fl_start_time"])
sep_pret["fl_peak_time"] = pd.to_datetime(sep_pret["fl_peak_time"])
sep_pret["cme_1st_app_time"] = pd.to_datetime(sep_pret["cme_1st_app_time"])
sep_pret["fl_end_time"] = pd.to_datetime(sep_pret["fl_end_time"])
##int
sep_pret["lasco_linear_speed"] = sep_pret["lasco_linear_speed"].astype("Int64")
sep_pret["lasco_cme_width"] = sep_pret["lasco_cme_width"].astype("Int64")
sep_pret["daily_sn"] = sep_pret["daily_sn"].astype("Int64")
sep_pret["AR_Area"] = sep_pret["AR_Area"].astype("Int64")
sep_pret["fl_lon"] = sep_pret["fl_lon"].astype("Int64")
sep_pret["fl_lat"] = sep_pret["fl_lat"].astype("Int64")
sep_pret["AR_long"] = sep_pret["AR_long"].astype("Int64")
sep_pret["AR_lat"] = sep_pret["AR_lat"].astype("Int64")
sep_pret["AR_Area"] = sep_pret["AR_Area"].astype("Int64")
##float
sep_pret["fl_goes_xray"] = sep_pret["fl_goes_xray"].astype(float)
#str
sep_pret["AR_Mcintosh"] = sep_pret["AR_Mcintosh"].astype("str")
sep_pret["AR_Hale"] = sep_pret["AR_Hale"].astype("str")


# longitude saturation
#giving same location values for behind limb flares (lon only neccesary)
sep_pret["fl_lon"] = sep_pret["fl_lon"].clip(upper=90).clip(lower=-90)
sep_pret["AR_long"] = sep_pret["fl_lon"].clip(upper=90).clip(lower=-90)


# correcting wrong datetime
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 52767, 'fl_peak_time'] = pd.Timestamp('2002-03-31 03:17:00') #change of time
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 52767, 'fl_end_time'] = pd.Timestamp('2002-03-31 03:28:00')  #change of time
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 55471, 'fl_end_time'] = pd.Timestamp('2003-03-30 03:02:00')  #change of time
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 70316, 'fl_peak_time'] = pd.Timestamp('2011-04-12 00:00:00') #NOAA midnight convention
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 72209, 'fl_end_time'] = pd.Timestamp('2012-03-25 03:20:00')  #change of time
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 72256, 'fl_peak_time'] = pd.Timestamp('2012-04-06 00:00:00') #NOAA midnight convention
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 72662, 'fl_peak_time'] = pd.Timestamp('2012-06-07 00:00:00') #NOAA midnight convention
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 73080, 'fl_peak_time'] = pd.Timestamp('2012-07-31 00:00:00') #NOAA midnight convention
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 73548, 'fl_peak_time'] = pd.Timestamp('2012-10-14 00:00:00') #NOAA midnight convention
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 32868943, 'fl_peak_time'] = pd.Timestamp('2014-05-26 00:00:00') #NOAA midnight convention
sep_pret.loc[sep_pret['noaa_flares_hec_id'] == 32873587, 'fl_peak_time'] = pd.Timestamp('2017-03-27 00:00:00') #NOAA midnight convention
sep_pret.loc[255, "fl_start_time"] = pd.NaT


#diff times (cme & flares)
#/!\ always in minutes
sep_pret["fl_rising_time"] = (sep_pret["fl_peak_time"] - sep_pret["fl_start_time"]) / pd.Timedelta(minutes=1)
sep_pret["cme_rising_time"] = (sep_pret["cme_1st_app_time"] - sep_pret["fl_start_time"]) / pd.Timedelta(minutes=1)
sep_pret["fl_total_time"] = (sep_pret["fl_end_time"] - sep_pret["fl_start_time"]) / pd.Timedelta(minutes=1)


#encoding Hale & Mcintosh classification
sep_pret['AR_Hale'] = sep_pret['AR_Hale'].replace(['None', 'none', 'NaN', None], np.nan) 
sep_pret['AR_Mcintosh'] = sep_pret['AR_Mcintosh'].replace(['None', 'none', 'NaN', None], np.nan)

codes, uniques = pd.factorize(sep_pret['AR_Hale'], use_na_sentinel=True) #putting a -1 classification for NaN value
sep_pret['AR_Hale_int'] = np.where(codes == -1, np.nan, codes + 1)       #convert the -1 classification into np.nan

codes, uniques = pd.factorize(sep_pret['AR_Mcintosh'], use_na_sentinel=True) 
sep_pret['AR_Mcintosh_int'] = np.where(codes == -1, np.nan, codes + 1)


# Index and chronological sort
##putting the variables in order and removing the useless ones
sep_pret = sep_pret[[#outputs
                     'GSEP flag', '>= S1', '>= S2', '>= S3', '= S1', '= S2', '= S3', '= S4', 'S_class', 'noaa_pf10MeV',  
                     #flares 
                     'fl_start_time', 'fl_rising_time', 'fl_total_time', 'fl_goes_xray', 'fl_lon', 'fl_lat', 
                     #flares count & sum
                     'noaa_flares_flares_count_last24h', 'noaa_flares_xray_sum_last24h',   
                     'noaa_flares_AR_flares_count_last24h', 'noaa_flares_AR_xray_sum_last24h', 
                     'noaa_flares_flares_count_last48h', 'noaa_flares_xray_sum_last48h',   #flares
                     'noaa_flares_AR_flares_count_last48h', 'noaa_flares_AR_xray_sum_last48h',
                     #ARs
                     'AR_long', 'AR_lat', 'AR_Area', 'AR_Hale', 'AR_Hale_int', 'AR_Mcintosh', 'AR_Mcintosh_int', 
                     #CME
                     'cme_rising_time', 'lasco_linear_speed', 'lasco_cme_width', 
                     #SN
                     'daily_sn' 
                     ]]

sep_pret.insert(0, 'SEP PRET index', sep_pret.index) #adding the index: (0 to 268 are (G)SEP events; the rest are quiet events)
sep_pret = sep_pret.sort_values("fl_start_time") # sort data by date, no more by flag

#saving
#sep_pret.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v3.pkl")

#plot the histograms repartition of the parameters
# plot_repartition(sep_pret)

del codes, uniques
#%% UNDERSAMPLING (SEP PRET reduced)

#if you need to load the entire dataset:
# sep_pret = dataset_reading.load_sep_pret_v3()

sep_pret_reduced = sep_pret.copy()

# Big 3 thresholds : conditions must be True OR value must be NaN/missing
## CME width
sep_pret_reduced = sep_pret_reduced[
    (sep_pret_reduced["lasco_cme_width"] > 110)     #110 degrees
    | (sep_pret_reduced["lasco_cme_width"].isna())
]

## CME speed
sep_pret_reduced = sep_pret_reduced[
    (sep_pret_reduced["lasco_linear_speed"] > 850)  #850 km/s
    | (sep_pret_reduced["lasco_linear_speed"].isna())
]

# flare X-ray flux (recording by GOES)
sep_pret_reduced = sep_pret_reduced[
    (sep_pret_reduced["fl_goes_xray"] > 5e-6)        #C5.0
    | (sep_pret_reduced["fl_goes_xray"].isna())
]

#addding back the event 96: C5 flares that lead to S2 SEP storm
event_96 = sep_pret[sep_pret["SEP PRET index"] == 96] 

sep_pret_reduced = (
    pd.concat([sep_pret_reduced, event_96])
    .drop_duplicates()
    .sort_values(by="fl_start_time")
)

#all event removed during the undersampling process:
sep_pret_removed = sep_pret.loc[~sep_pret.index.isin(sep_pret_reduced.index)]

#saving
#sep_pret_reduced.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v3_reduced.pkl")

del event_96
#%% OVERSAMPLING (SEP PRET sampled)

#if you need to load the undersampled dataset:
# sep_pret_reduced = dataset_reading.load_sep_pret_v3_reduced()

X = sep_pret_reduced.drop(columns='>= S1'); y = sep_pret_reduced[['>= S1']]

oversample_03 = RandomOverSampler(sampling_strategy=2*0.03) #the minority class represents 6% of the size of the majority class.
oversample_50 = RandomOverSampler(sampling_strategy=1)     #/!\ high risk of overfitting

X_sample_03, y_sample_03 = oversample_03.fit_resample(X, y)
X_sample_50, y_sample_50 = oversample_50.fit_resample(X, y)

sep_pret_sample_03 = X_sample_03
sep_pret_sample_03.insert(1, ">= S1", y_sample_03['>= S1']) #2nd position
sep_pret_sample_50 = X_sample_50
sep_pret_sample_50.insert(1, ">= S1", y_sample_50['>= S1']) #2nd position

#saving 
# sep_pret_sample_03.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v3_sample_03.pkl")
# sep_pret_sample_50.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v3_sample_50.pkl")

del X, X_sample_03, X_sample_50, y, y_sample_03, y_sample_50, oversample_03, oversample_50
