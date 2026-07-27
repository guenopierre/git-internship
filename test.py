import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

from usefull_functions import correlation_matrix, run_pca, run_all_combinations
from GSEP_extended import build_GSEP_int_extended, build_GSEP_extended
import dataset_reading


GSEP = build_GSEP_extended()
GSEP_int = build_GSEP_int_extended()

srs_combine_complete_corrected = dataset_reading.load_srs_combine_complete_corrected()


#%%

columns_1 = ['lasco_linear_speed', 'fl_lon', 'fl_lat', 'noaa_pf10MeV', 'fluence_gt10MeV']
corr_matrix_pearson = correlation_matrix(GSEP_int, columns_1, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True, title='Correlation matrix')

#%%

pca, GSEP_pca = run_pca(GSEP_int[columns_1], correlation_circle=True)   


#%%

inputs_df = GSEP_int
all_inputs = ['daily_sn', 'AR_Area', 'AR_Mcintosh_ranked', 'AR_Hale_int_ranked', 'AR_Mcintosh_Z_int_ranked', 
'AR_Mcintosh_p_int_ranked', 'AR_Mcintosh_c_int_ranked', 'AR_Mcintosh_length_int_ranked', 'AR_Mcintosh_penumbra_type_int_ranked']
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


