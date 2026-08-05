import dataset_reading

sep_pret_ml = dataset_reading.load_sep_pret()

sep_pret_ml = sep_pret_ml.dropna()


from usefull_functions import run_pca, run_ml_binary_classification_sklearn, find_combi, correlation_matrix


#%% PCA

# input_output_pca = sep_pret_ml.drop(columns=['>= S1', '>= S2', '>= S3', '= S1', '= S2', '= S3', '= S4', 'S_class'])
input_output_pca = sep_pret_ml[['GSEP flag', 'fl_total_time', 'fl_goes_xray', 'daily_sn', 'fl_lon',
'fl_rising_time', 'lasco_linear_speed', 'cme_rising_time', 'noaa_flares_AR_xray_max_last24h',
'lasco_cme_width', 'AR_Area', 'noaa_flares_AR_flares_count_last24h']]


_, _, _ = run_pca(input_output_pca, n_components=3, correlation_circle=True)
#%% ML              

inputs_df = sep_pret_ml[['fl_total_time', 'fl_goes_xray', 'fl_lon', 'noaa_flares_AR_flares_count_last24h', 
 'AR_Area', 'lasco_cme_width', 'daily_sn'
]]

output_df = sep_pret_ml[['GSEP flag']]

run_ml_binary_classification_sklearn(
    inputs_df,
    output_df,
    pca_n_comp = 0
)

#%%

comb_name = find_combi(inputs_df, ["fl_rising_time", "fl_total_time", "cme_rising_time"])

#%%

correlation_matrix(sep_pret_ml.drop(columns=['>= S1', '>= S2', '>= S3', '= S1', '= S2', '= S3', '= S4', 'S_class', 'noaa_pf10MeV']), sep_pret_ml.drop(columns=['>= S1', '>= S2', '>= S3', '= S1', '= S2', '= S3', '= S4', 'S_class', 'noaa_pf10MeV']).columns.tolist(), 
                   title='SEP-PRET 1.0')