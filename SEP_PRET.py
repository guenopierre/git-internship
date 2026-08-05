#%% librairies and datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from itertools import chain

import dataset_reading # personal loading datasets file

noaa_flares = dataset_reading.load_noaa_flares_extended() # personal extended version
cdaw_cme = dataset_reading.load_cdaw_cme() #original version
GSEP = dataset_reading.load_GSEP_extended() #personal extended version
PRISM_1h = dataset_reading.load_PRISM_analyzed_rolling_combinded_seq_1hours() #original version
PRISM_24h = dataset_reading.load_PRISM_analyzed_rolling_combinded_seq_24hours() #original version

# crop datasets on the same period 
date_a = pd.Timestamp('1986-02-04') #GSEP & PRISM beginning
# not the most restrictive one (cdaw start in 1996) --> no cme informations on the first sep events
date_b = pd.Timestamp('2017-09-10') #GSEP end (most restrictive one here)
noaa_flares = noaa_flares[noaa_flares['time_start'].between(date_a, date_b)]

del date_a, date_b #not more usefull
#%% columns

GSEP_col = GSEP.columns.tolist()
#CME related
cme_col = ['cme_1st_app_time', 'lasco_linear_speed', 'lasco_cme_width']
#Flare related
flare_col = ['noaa_flares_hec_id', 'fl_start_time', 'fl_end_time', 'fl_peak_time', 'fl_lon', 'fl_lat', 'fl_goes_xray', 
             'noaa_flares_flares_count_last24h', 'noaa_flares_xray_average_last24h', 'noaa_flares_xray_max_last24h', 
             'noaa_flares_AR_flares_count_last24h', 'noaa_flares_AR_xray_average_last24h', 'noaa_flares_AR_xray_max_last24h']
#Active Region
AR_col = ['noaa_ar', 'AR_long', 'AR_lat', 'AR_Area', 'AR_Mcintosh' , 'AR_Hale']
#Sunspot Numbers 
SN_col = ['daily_sn']
#Proton flux ++
proton_flux_col = ['noaa_pf10MeV']
#Flags (S-Storm class)
flags_col = ['>= S1', '>= S2', '>= S3', '= S1', '= S2', '= S3', '= S4', 'S_class']

#%% GSEP "active days"

sep_pret_active = pd.DataFrame()
#copying all intersting parameters from GSEP (extended)
sep_pret_active = GSEP[list(chain.from_iterable([cme_col, flare_col, AR_col, SN_col, proton_flux_col, flags_col]))]

# active day flag (not always >= S1, see GSEP SEP definition)
sep_pret_active['GSEP flag'] = 1

del GSEP_col, cme_col, flare_col, flags_col, AR_col, SN_col, proton_flux_col #not more usefull
#%% noaa flares "quiet days"

#remove the flares that have led to SEP, using the hec_id parameter (has been added to GSEP with my extension)   
noaa_flares_quiet = noaa_flares[~noaa_flares['hec_id'].isin(sep_pret_active['noaa_flares_hec_id'])]

#rename parameters to keep the same names
sep_pret_quiet =  noaa_flares_quiet.rename(columns={
    'hec_id' : 'noaa_flares_hec_id',
    'time_start': 'fl_start_time', 
    'time_peak': 'fl_peak_time',
    'time_end': 'fl_end_time', 
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

#putting the flags to 0 (even noaa_pf10meV)
sep_pret_quiet['>= S1'] = 0; sep_pret_quiet['>= S2'] = 0; sep_pret_quiet['>= S3'] = 0
sep_pret_quiet['= S1'] = 0; sep_pret_quiet['= S2'] = 0; sep_pret_quiet['= S3'] = 0; sep_pret_quiet['= S4'] = 0
sep_pret_quiet['S_class'] = 0; sep_pret_quiet['GSEP flag'] = 0; sep_pret_quiet['noaa_pf10MeV'] = 0

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

#adding the Active Region informations (Mcintosh, Hale, Location, Lat, Lon, Area)
sep_pret_quiet = add_ar_info(sep_pret_quiet) #/!\ takes a while
#removing the str location information, we converted into int lat and lon columns 
sep_pret_quiet = sep_pret_quiet.drop(columns=['AR_Location'])
#removing AR parameters we don't need (Lo, NN, LL)
sep_pret_quiet = sep_pret_quiet.drop(columns=["AR_LL", "AR_Lo", "AR_NN"])
#removing flare useless information

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

#adding the CME informations (t_start, v_lin, width)
sep_pret_quiet = merge_cme_info(sep_pret_quiet, cdaw_cme) #/!\ takes a while
#we rename them with GSEP convention names
sep_pret_quiet =  sep_pret_quiet.rename(columns={
    't_start' : 'cme_1st_app_time',
    'v_lin': 'lasco_linear_speed', 
    'width': 'lasco_cme_width'
})

#%% Concatenation

#concatenation of active and quiet days, removing non-common columns
sep_pret = pd.concat([sep_pret_active, sep_pret_quiet], join='inner', ignore_index=True)

#%% adjustments

#giving same location values for behind limb flares (lon only neccesary)
sep_pret["fl_lon"] = sep_pret["fl_lon"].clip(upper=90).clip(lower=-90)
sep_pret["AR_long"] = sep_pret["fl_lon"].clip(upper=90).clip(lower=-90)

#correct type conversion
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
sep_pret["noaa_flares_flares_count_last24h"] = sep_pret["noaa_flares_flares_count_last24h"].astype("Int64")
sep_pret["noaa_flares_AR_flares_count_last24h"] = sep_pret["noaa_flares_AR_flares_count_last24h"].astype("Int64")
##float
sep_pret["fl_goes_xray"] = sep_pret["fl_goes_xray"].astype(float)

#sort data by date, no more by flag
sep_pret = sep_pret.sort_values("fl_start_time")

#diff times (cme & flares)
##/!\ always in minutes
sep_pret["fl_rising_time"] = (sep_pret["fl_peak_time"] - sep_pret["fl_start_time"]) / pd.Timedelta(minutes=1)
sep_pret["cme_rising_time"] = (sep_pret["cme_1st_app_time"] - sep_pret["fl_start_time"]) / pd.Timedelta(minutes=1)
sep_pret["fl_total_time"] = (sep_pret["fl_end_time"] - sep_pret["fl_start_time"]) / pd.Timedelta(minutes=1)

#%% ML tailored dataset

#copy existing dataset
sep_pret_ml = sep_pret.copy()

#remove useless variables (not informative for ML)
##datetime 
sep_pret_ml = sep_pret_ml.drop(columns=["fl_start_time", "fl_peak_time", "fl_end_time", 
                                        "cme_1st_app_time", "noaa_flares_hec_id"])
##AR number 
sep_pret_ml = sep_pret_ml.drop(columns=["noaa_ar"])

#convert str values to int, and remove str
##Hale
sep_pret_ml['AR_Hale_int'] = pd.factorize(sep_pret_ml['AR_Hale'])[0] + 1 #to start at 0
sep_pret_ml = sep_pret_ml.drop(columns=["AR_Hale"])
##McIntosh
sep_pret_ml['AR_Mcintosh_int'] = pd.factorize(sep_pret_ml['AR_Mcintosh'])[0] +1 #to start at 0
sep_pret_ml = sep_pret_ml.drop(columns=["AR_Mcintosh"])

def clean_missing_values(df):
    """
    Convert common missing-value representations in a DataFrame to np.nan.
    """
    missing_values = ['', ' ', 'NA', 'N/A', 'na', 'n/a', 'null',
                       'NULL', 'None', 'none', '-', '--', '?', 'missing', 
                       '<NA>']
    
    return df.replace(missing_values, np.nan)
sep_pret = clean_missing_values(sep_pret) 

#%% Adjusting parameters position

sep_pret = sep_pret[['GSEP flag', '>= S1', '>= S2', '>= S3', '= S1', '= S2', '= S3', '= S4', 'S_class', 'noaa_pf10MeV',  #flags
                     'noaa_flares_hec_id', 'fl_start_time', 'fl_peak_time', 'fl_end_time', 'fl_rising_time', 'fl_total_time', 'fl_goes_xray', 'fl_lon', 'fl_lat', 
                     'noaa_flares_flares_count_last24h', 'noaa_flares_xray_average_last24h', 'noaa_flares_xray_max_last24h',  #flares
                     'noaa_flares_AR_flares_count_last24h', 'noaa_flares_AR_xray_average_last24h', 'noaa_flares_AR_xray_max_last24h', 
                     'noaa_ar', 'AR_long', 'AR_lat', 'AR_Area', 'AR_Hale', 'AR_Mcintosh', #ARs
                     'cme_1st_app_time', 'cme_rising_time', 'lasco_linear_speed', 'lasco_cme_width', #cme
                     'daily_sn' #SN
                     ]]

sep_pret_ml = sep_pret_ml[['GSEP flag', '>= S1', '>= S2', '>= S3', '= S1', '= S2', '= S3', '= S4', 'S_class', 'noaa_pf10MeV',  #flags
                     'fl_rising_time', 'fl_total_time', 'fl_goes_xray', 'fl_lon', 'fl_lat', 
                     'noaa_flares_flares_count_last24h', 'noaa_flares_xray_average_last24h', 'noaa_flares_xray_max_last24h',  #flares
                     'noaa_flares_AR_flares_count_last24h', 'noaa_flares_AR_xray_average_last24h', 'noaa_flares_AR_xray_max_last24h', 
                     'AR_long', 'AR_lat', 'AR_Area', 'AR_Hale_int', 'AR_Mcintosh_int', #ARs
                     'cme_rising_time', 'lasco_linear_speed', 'lasco_cme_width', #cme
                     'daily_sn' #SN
                     ]]

#%% plot repartition parameters 

def plot_repartition(df, bins=100, figsize=(8, 5), top_n=100,
                      gsep_col="GSEP flag", column_params=None):
    """
    Affiche la répartition (distribution) de chaque colonne d'un dataframe.
 
    - Colonnes "flag" (peu de valeurs discrètes, ex 0/1/2/3/4) -> diagramme en barres
    - Colonnes numériques continues -> histogramme
    - Colonnes non numériques -> diagramme en barres des value_counts (top_n catégories)
 
    Si `gsep_col` est présente dans le dataframe, chaque colonne (sauf gsep_col
    elle-même) est scindée en deux sous-ensembles superposés sur le même graphique :
    - gsep_col == 0 -> bleu, à l'arrière-plan
    - gsep_col == 1 -> rouge, au premier plan (plus opaque, dessiné par-dessus)
 
    Parameters
    ----------
    df : pd.DataFrame
    bins : int
        Nombre de bins par défaut pour les histogrammes (utilisé si la colonne
        n'a pas d'override dans `column_params`).
    figsize : tuple
    top_n : int
        Nombre de catégories conservées pour les colonnes non numériques.
    gsep_col : str
        Nom de la colonne flag (0/1) utilisée pour scinder les données.
    column_params : dict, optionnel
        Réglages par colonne. Clés reconnues : "xscale", "yscale", "bins", "normalize".
        - "xscale" / "yscale" : 'linear', 'log', ... (passé tel quel à plt.xscale/yscale)
        - "bins" : int ou séquence de bornes, remplace le `bins` global pour cette colonne
        - "normalize" : bool -> si True, les valeurs sont exprimées en pourcentage du
          temps (somme ~100 par sous-ensemble) plutôt qu'en count brut
 
        Exemple :
        {
            "ma_colonne": {"xscale": "log", "yscale": "log", "bins": 50, "normalize": True},
            "autre_col": {"normalize": True},
        }
    """
    def _get_bin_edges(data, bins, xscale):
        """
        Calcule des bornes de bins communes (linéaires ou log) pour pouvoir superposer
        deux histogrammes (flag=0 / flag=1) sur des bins strictement identiques.
        Si `bins` est déjà une séquence de bornes, elle est renvoyée telle quelle.
        """
        if hasattr(bins, "__len__"):
            return bins  # bornes explicites déjà fournies
     
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
        """
        Trie une Series de value_counts par index croissant si l'index est numérique
        (ou convertible en nombre). Si ce n'est pas le cas (vraies catégories textuelles),
        on garde l'ordre existant (par fréquence décroissante).
        """
        try:
            numeric_idx = pd.to_numeric(counts.index)
            order = np.argsort(numeric_idx.to_numpy())
            return counts.iloc[order]
        except (ValueError, TypeError):
            return counts
    column_params = column_params or {}
    has_split = gsep_col in df.columns
    if has_split:
        df0 = df[df[gsep_col] == 0]
        df1 = df[df[gsep_col] == 1]
 
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
 
        # sous-ensembles à tracer : superposition bleu (flag=0, arrière-plan) / rouge (flag=1, premier plan)
        if has_split and col != gsep_col:
            subsets = [
                (df0[col], "tab:blue", 0.5, f"{gsep_col} = 0", 2),
                (df1[col], "tab:red", 0.75, f"{gsep_col} = 1", 3),
            ]
        else:
            subsets = [(series, "tab:blue", 0.85, None, 2)]
 
        plotted_labels = []
 
        if is_flag:
            all_vals = sorted(set(series.dropna().unique()))
            widths = [0.55] if len(subsets) == 1 else [0.55, 0.28]
 
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
            # catégories déterminées sur la colonne entière pour rester alignées entre les 2 sous-ensembles
            top_categories = series.value_counts(dropna=True).head(top_n).index
            widths = [0.8] if len(subsets) == 1 else [0.8, 0.42]
 
            for (data, color, alpha, label, zorder), width in zip(subsets, widths):
                clean = data.dropna()
                counts = clean.value_counts().reindex(top_categories, fill_value=0)
                counts = _sort_counts_increasing(counts)  # index croissant si numérique
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
        plt.show()

plot_repartition(sep_pret_ml, column_params=
                 {"lasco_linear_speed": {"xscale":"log","yscale": "log"},
                  "lasco_cme_width": {"xscale":"log","yscale": "log"},
                  "fl_total_time": {"xscale":"log","yscale": "log"},
                  "fl_lon": {"yscale": "log", "bins" : 50}, 
                  "fl_lat": {"yscale": "log", "bins" : 50}, 
                  "fl_goes_xray": {"yscale": "log", "xscale" : "log"},
                  "noaa_flares_flares_count_last24h": {"yscale": "log"}, 
                  "noaa_flares_xray_average_last24h": {"yscale": "log", "xscale" : "log"}, 
                  "noaa_flares_xray_max_last24h": {"yscale": "log", "xscale" : "log"}, 
                  "noaa_flares_AR_flares_count_last24h": {"yscale": "log"}, 
                  "noaa_flares_AR_xray_average_last24h": {"yscale": "log", "xscale" : "log"}, 
                  "noaa_flares_AR_xray_max_last24h": {"yscale": "log", "xscale" : "log"},
                  "AR_long" : {"yscale":"log"}, 
                  "AR_lat" : {"yscale":"log"}, 
                  "AR_Area" : {"yscale":"log"}, 
                  "AR_Mcintosh_int" : {"yscale":"log"}, 
                  "AR_Hale_int" : {"yscale":"log"}, 
                  "daily_sn" : {"yscale":"log"}, 
                  "noaa_pf10MeV" : {"yscale":"log"}, 
                  "fl_rising_time": {"yscale": "log", "xscale" : "log"}, 
                  "cme_rising_time": {"yscale": "log", "xscale" : "log"}})
                  
#%% pca visualisation
from usefull_functions import run_pca

pca, SEP_Pret_pca = run_pca(sep_pret_ml, correlation_circle=True) 

#%%

sep_pret_ml.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/sep_pret_ml.pkl")
