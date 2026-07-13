import numpy as np
import pandas as pd
from dataset_reading import GSEP_list, formal_sep



#%% functions

def formal_sep_filtering(timestamp, cdaw_start, cdaw_max, cme_id,
                         cme_launch_time, cme_1st_app_time,lasco_cme_width, 
                         p_cme_width, lasco_linear_speed,p_cme_speed, 
                         fl_start_time, fl_peak_time, noaa_pf10MeV, ppf_gt10MeV,
                         slice_start, slice_end, 
                         slice_choice = False, window=48, repartition=[2/3, 1/3]):
    if len(repartition) != 2:
        print('repartition argument is not at good size')
        return None
    else:
        if repartition[0] + repartition[1] != 1.0:
            print('be careful, repartition is not =1 ')
        
        
        
        formal_filtered = {}
        # On itère sur timestamp ET cdaw_start en parallèle
        
        if slice_choice == False:
            for t, cstart, cmax, cme_id, cme_launch_time, cme_1st_app_time, lasco_cme_width, p_cme_width, lasco_linear_speed, p_cme_speed, fl_start_time, fl_peak_time, noaa_pf10MeV, ppf_gt10MeV, slice_start, slice_end in zip(timestamp, cdaw_start, cdaw_max, cme_id, cme_launch_time ,cme_1st_app_time, lasco_cme_width, p_cme_width, lasco_linear_speed, p_cme_speed, fl_start_time, fl_peak_time, noaa_pf10MeV, ppf_gt10MeV, slice_start, slice_end):
                start = pd.Timestamp(t) - pd.Timedelta(hours=window * repartition[0])
                end = pd.Timestamp(t) + pd.Timedelta(hours=window * repartition[1])
                df_filtered = formal_sep.loc[start:end].copy()
                
                # Ajout de la colonne avec la valeur répétée sur toutes les lignes
                df_filtered['GSEP_timestamp'] = t
                df_filtered['cdaw_start_time'] = cstart
                df_filtered['cdaw_max_time'] = cmax
                df_filtered['cme_id'] = cme_id
                df_filtered['cme_launch_time'] = cme_launch_time
                df_filtered['cme_1st_app_time'] = cme_1st_app_time
                df_filtered['lasco_cme_width'] = lasco_cme_width
                df_filtered['p_cme_width'] = p_cme_width
                df_filtered['lasco_linear_speed'] = lasco_linear_speed
                df_filtered['p_cme_speed'] = p_cme_speed
                df_filtered['fl_start_time'] = fl_start_time
                df_filtered['fl_peak_time'] = fl_peak_time
                df_filtered['noaa_pf10MeV'] = noaa_pf10MeV
                df_filtered['ppf_gt10MeV'] = ppf_gt10MeV
                df_filtered['slice_start'] = slice_start
                df_filtered['slice_end'] = slice_end
                
                formal_filtered[t] = df_filtered
        else:
            for t, cstart, cmax, cme_id, cme_launch_time, cme_1st_app_time, lasco_cme_width, p_cme_width, lasco_linear_speed, p_cme_speed, fl_start_time, fl_peak_time, noaa_pf10MeV, ppf_gt10MeV, slice_start, slice_end in zip(timestamp, cdaw_start, cdaw_max, cme_id, cme_launch_time ,cme_1st_app_time, lasco_cme_width, p_cme_width, lasco_linear_speed, p_cme_speed, fl_start_time, fl_peak_time, noaa_pf10MeV, ppf_gt10MeV, slice_start, slice_end):
                start = pd.Timestamp(slice_start) 
                end = pd.Timestamp(slice_end)
                df_filtered = formal_sep.loc[start:end].copy()
                
                # Ajout de la colonne avec la valeur répétée sur toutes les lignes
                df_filtered['GSEP_timestamp'] = t
                df_filtered['cdaw_start_time'] = cstart
                df_filtered['cdaw_max_time'] = cmax
                df_filtered['cme_id'] = cme_id
                df_filtered['cme_launch_time'] = cme_launch_time
                df_filtered['cme_1st_app_time'] = cme_1st_app_time
                df_filtered['lasco_cme_width'] = lasco_cme_width
                df_filtered['p_cme_width'] = p_cme_width
                df_filtered['lasco_linear_speed'] = lasco_linear_speed
                df_filtered['p_cme_speed'] = p_cme_speed
                df_filtered['fl_start_time'] = fl_start_time
                df_filtered['fl_peak_time'] = fl_peak_time
                df_filtered['noaa_pf10MeV'] = noaa_pf10MeV
                df_filtered['ppf_gt10MeV'] = ppf_gt10MeV
                df_filtered['slice_start'] = slice_start
                df_filtered['slice_end'] = slice_end
                
                formal_filtered[t] = df_filtered

        return formal_filtered


def sep_dictionary_function(window = 48, repartition = [1/3,2/3], extension = False, slice_choice = False):
    sep_dictionary = formal_sep_filtering(GSEP_list['timestamp'], GSEP_list['cdaw_start_time'], GSEP_list['cdaw_max_time'],
                                          GSEP_list['cme_id'], GSEP_list['cme_launch_time'], GSEP_list['cme_1st_app_time'], 
                                          GSEP_list['lasco_cme_width'], GSEP_list['p_cme_width'], GSEP_list['lasco_linear_speed'], 
                                          GSEP_list['p_cme_speed'], GSEP_list['fl_start_time'], GSEP_list['fl_peak_time'],
                                          GSEP_list['noaa_pf10MeV'], GSEP_list['ppf_gt10MeV'], 
                                          GSEP_list['slice_start'], GSEP_list['slice_end'], 
                                          window=window, repartition = repartition, slice_choice=slice_choice)
    
    if extension == False:
        print('sep_dictionary without extension')
        return sep_dictionary
    
    elif extension == True:
        for date_key, dataframe in sep_dictionary.items():
            col_name = 'xrayl'
            max_col_name = 'max_x_ray'
            time_max_col_name = 'time_max_x_ray'
            
            if dataframe[col_name].dropna().empty:
                dataframe[max_col_name] = 0
                dataframe[time_max_col_name] = np.nan
            else:
                dataframe[max_col_name] = dataframe[col_name].max()
                dataframe[time_max_col_name] = dataframe[col_name].idxmax()
            
            
            for i in range(1, 12):
                col_name = f'F{i}'
                max_col_name = f'max_F{i}'
                time_max_col_name = f'time_max_F{i}'
        
                # Cas où la colonne est vide (ou entièrement NaN)
                if dataframe[col_name].dropna().empty:
                    dataframe[max_col_name] = 0
                    dataframe[time_max_col_name] = np.nan
                else:
                    dataframe[max_col_name] = dataframe[col_name].max()
                    dataframe[time_max_col_name] = dataframe[col_name].idxmax()
        
        return sep_dictionary
    
    else:
        print('extension argument should be bool')
        
#%% timestamp


#---- global
sep_dictionary = sep_dictionary_function(extension=True)


#--- cdaw
cdaw_sep_dictionary = {
    cle: df
    for cle, df in sep_dictionary.items()
    if not df.empty and pd.notna(df['cdaw_start_time'].iloc[0])
}

#--- noaa
noaa_sep_dictionary = {
    cle: df
    for cle, df in sep_dictionary.items()
    if not df.empty and pd.notna(df['noaa_pf10MeV'].iloc[0])
}

#---- cme
cme_sep_dictionary = {
    cle: df
    for cle, df in sep_dictionary.items()
    if not df.empty and pd.notna(df['cme_1st_app_time'].iloc[0])
} 

#---- complete (cdaw + cme + flare)
complete_sep_dictionary =  {
    cle: df
    for cle, df in sep_dictionary.items()
    if not df.empty and pd.notna(df['cdaw_start_time'].iloc[0]) and pd.notna(df['cme_1st_app_time'].iloc[0]) and pd.notna(df['fl_start_time'].iloc[0])
}

#---- best (hand selected)
cles = list(complete_sep_dictionary.keys())
selected_indices =  [0, 1, 2, 3, 4, 8, 10, 11, 14, 15, 17, 18, 20, 21, 25, 31, 32, 39, 40, 46, 47, 48, 49, 53, 54, 57, 61, 62, 64, 68, 71, 73, 77, 78, 80, 82, 87, 89, 93, 94, 95, 96, 98, 101, 106, 108, 110, 114]
selected_keys = [cles[i] for i in selected_indices]
best_sep_dictionary = {k: complete_sep_dictionary[k] for k in selected_keys}



#---- S0, S1, etc. 
S0_threshold = 10; S1_threshold = 10**2; S2_threshold = 10**3; S3_threshold = 10**4;

S0_ZPGT10W_CORR_sep_dictionary, S1_ZPGT10W_CORR_sep_dictionary, S2_ZPGT10W_CORR_sep_dictionary, S3_ZPGT10W_CORR_sep_dictionary, S4_ZPGT10W_CORR_sep_dictionary = {}, {}, {}, {}, {}
S0_noaa_pf10MeV_sep_dictionary, S1_noaa_pf10MeV_sep_dictionary, S2_noaa_pf10MeV_sep_dictionary, S3_noaa_pf10MeV_sep_dictionary, S4_noaa_pf10MeV_sep_dictionary = {}, {}, {}, {}, {}
S0_ppf_gt10MeV_sep_dictionary, S1_ppf_gt10MeV_sep_dictionary, S2_ppf_gt10MeV_sep_dictionary, S3_ppf_gt10MeV_sep_dictionary, S4_ppf_gt10MeV_sep_dictionary = {}, {}, {}, {}, {}


for key, df in sep_dictionary.items():
    max_val = df['ZPGT10W_CORR'].max()
    
    if max_val <= S0_threshold:
        S0_ZPGT10W_CORR_sep_dictionary[key] = df
    elif max_val <= S1_threshold:
        S1_ZPGT10W_CORR_sep_dictionary[key] = df
    elif max_val <= S2_threshold:
        S2_ZPGT10W_CORR_sep_dictionary[key] = df
    elif max_val <= S3_threshold:
        S3_ZPGT10W_CORR_sep_dictionary[key] = df
    elif max_val > S3_threshold:  
        S4_ZPGT10W_CORR_sep_dictionary[key] = df
        
        
for key, df in sep_dictionary.items():
    max_val = df['noaa_pf10MeV'].max()
    
    if max_val <= S0_threshold:
        S0_noaa_pf10MeV_sep_dictionary[key] = df
    elif max_val <= S1_threshold:
        S1_noaa_pf10MeV_sep_dictionary[key] = df
    elif max_val <= S2_threshold:
        S2_noaa_pf10MeV_sep_dictionary[key] = df
    elif max_val <= S3_threshold:
        S3_noaa_pf10MeV_sep_dictionary[key] = df
    elif max_val > S3_threshold: 
        S4_noaa_pf10MeV_sep_dictionary[key] = df


for key, df in sep_dictionary.items():
    max_val = df['ppf_gt10MeV'].max()
    
    if max_val <= S0_threshold:
        S0_ppf_gt10MeV_sep_dictionary[key] = df
    elif max_val <= S1_threshold:
        S1_ppf_gt10MeV_sep_dictionary[key] = df
    elif max_val <= S2_threshold:
        S2_ppf_gt10MeV_sep_dictionary[key] = df
    elif max_val <= S3_threshold:
        S3_ppf_gt10MeV_sep_dictionary[key] = df
    elif max_val > S3_threshold:  
        S4_ppf_gt10MeV_sep_dictionary[key] = df


del S0_threshold, S1_threshold, S2_threshold, S3_threshold

#%% slice
#---- global

sep_dictionary_slice = sep_dictionary_function(extension=True, slice_choice=True)


#--- cdaw
cdaw_sep_dictionary_slice = {
    cle: df
    for cle, df in sep_dictionary_slice.items()
    if not df.empty and pd.notna(df['cdaw_start_time'].iloc[0])
}

#--- noaa
noaa_sep_dictionary_slice = {
    cle: df
    for cle, df in sep_dictionary_slice.items()
    if not df.empty and pd.notna(df['noaa_pf10MeV'].iloc[0])
}

#---- cme
cme_sep_dictionary_slice = {
    cle: df
    for cle, df in sep_dictionary_slice.items()
    if not df.empty and pd.notna(df['cme_1st_app_time'].iloc[0])
} 

#---- complete (cdaw + cme + flare)
complete_sep_dictionary_slice =  {
    cle: df
    for cle, df in sep_dictionary_slice.items()
    if not df.empty and pd.notna(df['cdaw_start_time'].iloc[0]) and pd.notna(df['cme_1st_app_time'].iloc[0]) and pd.notna(df['fl_start_time'].iloc[0])
}

#---- best (hand selected)
cles = list(complete_sep_dictionary_slice.keys())
selected_indices =  [0, 1, 2, 3, 4, 8, 10, 11, 14, 15, 17, 18, 20, 21, 25, 31, 32, 39, 40, 46, 47, 48, 49, 53, 54, 57, 61, 62, 64, 68, 71, 73, 77, 78, 80, 82, 87, 89, 93, 94, 95, 96, 98, 101, 106, 108, 110, 114]
selected_keys = [cles[i] for i in selected_indices]
best_sep_dictionary = {k: complete_sep_dictionary_slice[k] for k in selected_keys}



#---- S0, S1, etc. 
S0_threshold = 10; S1_threshold = 10**2; S2_threshold = 10**3; S3_threshold = 10**4;

S0_ZPGT10W_CORR_sep_dictionary_slice, S1_ZPGT10W_CORR_sep_dictionary_slice, S2_ZPGT10W_CORR_sep_dictionary_slice, S3_ZPGT10W_CORR_sep_dictionary_slice, S4_ZPGT10W_CORR_sep_dictionary_slice = {}, {}, {}, {}, {}
S0_noaa_pf10MeV_sep_dictionary_slice, S1_noaa_pf10MeV_sep_dictionary_slice, S2_noaa_pf10MeV_sep_dictionary_slice, S3_noaa_pf10MeV_sep_dictionary_slice, S4_noaa_pf10MeV_sep_dictionary_slice = {}, {}, {}, {}, {}
S0_ppf_gt10MeV_sep_dictionary_slice, S1_ppf_gt10MeV_sep_dictionary_slice, S2_ppf_gt10MeV_sep_dictionary_slice, S3_ppf_gt10MeV_sep_dictionary_slice, S4_ppf_gt10MeV_sep_dictionary_slice = {}, {}, {}, {}, {}


for key, df in sep_dictionary_slice.items():
    max_val = df['ZPGT10W_CORR'].max()
    
    if max_val <= S0_threshold:
        S0_ZPGT10W_CORR_sep_dictionary_slice[key] = df
    elif max_val <= S1_threshold:
        S1_ZPGT10W_CORR_sep_dictionary_slice[key] = df
    elif max_val <= S2_threshold:
        S2_ZPGT10W_CORR_sep_dictionary_slice[key] = df
    elif max_val <= S3_threshold:
        S3_ZPGT10W_CORR_sep_dictionary_slice[key] = df
    elif max_val > S3_threshold:  # max_val > 10^4
        S4_ZPGT10W_CORR_sep_dictionary_slice[key] = df
        
        
for key, df in sep_dictionary_slice.items():
    max_val = df['noaa_pf10MeV'].max()
    
    if max_val <= S0_threshold:
        S0_noaa_pf10MeV_sep_dictionary_slice[key] = df
    elif max_val <= S1_threshold:
        S1_noaa_pf10MeV_sep_dictionary_slice[key] = df
    elif max_val <= S2_threshold:
        S2_noaa_pf10MeV_sep_dictionary_slice[key] = df
    elif max_val <= S3_threshold:
        S3_noaa_pf10MeV_sep_dictionary_slice[key] = df
    elif max_val > S3_threshold: 
        S4_noaa_pf10MeV_sep_dictionary_slice[key] = df


for key, df in sep_dictionary_slice.items():
    max_val = df['ppf_gt10MeV'].max()
    
    if max_val <= S0_threshold:
        S0_ppf_gt10MeV_sep_dictionary_slice[key] = df
    elif max_val <= S1_threshold:
        S1_ppf_gt10MeV_sep_dictionary_slice[key] = df
    elif max_val <= S2_threshold:
        S2_ppf_gt10MeV_sep_dictionary_slice[key] = df
    elif max_val <= S3_threshold:
        S3_ppf_gt10MeV_sep_dictionary_slice[key] = df
    elif max_val > S3_threshold:  
        S4_ppf_gt10MeV_sep_dictionary_slice[key] = df


del S0_threshold, S1_threshold, S2_threshold, S3_threshold