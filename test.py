#%%
import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

from usefull_functions import correlation_matrix, run_pca, run_all_combinations
from GSEP_extended import build_GSEP_int_extended

GSEP_int = build_GSEP_int_extended(
    xray_flux= True,
    ar_info = True,
    sunspot_number = True,
    flags = True,
    slice_range = True,
    ref1 = True,
    ref2 = True
)
columns = GSEP_int.columns.tolist()

#%%

columns_1 = ['noaa_pf10MeV','daily_sn', 'AR_area', 'AR_z_int_ranked', 'AR_mag_type_int_ranked', 'group_configuration_int_ranked',
             'AR_z_penumbra_type_int_ranked', 'largest_spot_type_int_ranked', 'AR_z_length_int_ranked', 'spots_distribution_int_ranked']

corr_matrix_pearson = correlation_matrix(GSEP_int, columns_1, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True, title='configuration 2')

#%%

pca, GSEP_pca = run_pca(GSEP_int[columns_1], correlation_circle=True)


#%%

inputs_df = GSEP_int
all_inputs = ['daily_sn', 'AR_area', 'AR_z_int_ranked', 'AR_mag_type_int_ranked', 'group_configuration_int_ranked', 
'largest_spot_type_int_ranked', 'spots_distribution_int_ranked', 'AR_z_length_int_ranked', 'AR_z_penumbra_type_int_ranked']
outputs = GSEP_int['= S4']
result_file_path =  "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/220726_007.xlsx"


run_all_combinations(inputs_df, all_inputs, outputs, result_file_path,
                          model_choice='RandomForestClassifier',
                          model_params=[42, 'balanced'],
                          inputs_pca_nbr_pc=0,
                          test_size=0.2,
                          min_combo_size=1,
                          max_combo_size=None,
                          show_plot=False,
                          verbose=False)

#%%

import dataset_reading

PRISM_analyzed_rolling_combinded_seq_24hours = dataset_reading.load_PRISM_analyzed_rolling_combinded_seq_24hours()

PRISM_predictors = PRISM_analyzed_rolling_combinded_seq_24hours.columns.tolist()


