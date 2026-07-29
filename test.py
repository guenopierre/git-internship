import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

from usefull_functions import correlation_matrix, run_pca, run_all_combinations, run_ml_sep
from GSEP_extended import build_GSEP_int_extended, build_GSEP_extended
import dataset_reading


noaa = dataset_reading.load_noaa_flares_extended()
GSEP = build_GSEP_extended()
GSEP_int = build_GSEP_int_extended()
SN = dataset_reading.load_SN_d_tot_V2()

srs_combine_complete_corrected = dataset_reading.load_srs_combine_complete_corrected()


#%%

columns_1 = ['daily_sn','lasco_linear_speed', 'fl_lon', 'fl_lat', 'noaa_pf10MeV', 'fluence_gt10MeV']
corr_matrix_pearson = correlation_matrix(GSEP_int, columns_1, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True, title='Correlation matrix')

#%%

pca, GSEP_pca = run_pca(GSEP_int[columns_1], correlation_circle=True)   


#%%

run_ml_sep(GSEP[['daily_sn', 'AR_Mcintosh_int_ranked', 'AR_Hale_int_ranked']], 
           GSEP_int['>= S1'],
              model_choice='RandomForestClassifier', model_params=[42, 'balanced'])

#%%

run_ml_sep(GSEP[['AR_Mcintosh_Z_int', 'AR_Mcintosh_p_int', 'AR_Mcintosh_c_int']], 
           GSEP_int['>= S2'],
               model_choice='RandomForestClassifier', model_params=[42, 'balanced'])


#%% TO RUN AT HOME

# to do again

inputs_df = GSEP_int
colonnes = GSEP_int.columns.tolist()
all_inputs = ['daily_sn', 
'AR_Area',
'AR_Mcintosh_int_ranked',
'AR_Hale_int_ranked',
'AR_Mcintosh_Z_int_ranked',
'AR_Mcintosh_p_int_ranked',
'AR_Mcintosh_c_int_ranked',
'AR_Mcintosh_length_int_ranked',
'AR_Mcintosh_penumbra_type_int_ranked'
]

outputs = GSEP_int['>= S1']
result_file_path =  "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/280726_001.xlsx"
run_all_combinations(inputs_df, all_inputs, outputs, result_file_path,
                          model_choice='RandomForestClassifier',
                          model_params=[42, 'balanced'],
                          inputs_pca_nbr_pc=0,
                          test_size=0.2,
                          min_combo_size=1,
                          max_combo_size=None,
                          show_plot=False,
                          verbose=False)


outputs = GSEP_int['>= S2']
result_file_path =  "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/280726_002.xlsx"
run_all_combinations(inputs_df, all_inputs, outputs, result_file_path,
                          model_choice='RandomForestClassifier',
                          model_params=[42, 'balanced'],
                          inputs_pca_nbr_pc=0,
                          test_size=0.2,
                          min_combo_size=1,
                          max_combo_size=None,
                          show_plot=False,
                          verbose=False)

outputs = GSEP_int['>= S3']
result_file_path =  "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/280726_003.xlsx"
run_all_combinations(inputs_df, all_inputs, outputs, result_file_path,
                          model_choice='RandomForestClassifier',
                          model_params=[42, 'balanced'],
                          inputs_pca_nbr_pc=0,
                          test_size=0.2,
                          min_combo_size=1,
                          max_combo_size=None,
                          show_plot=False,
                          verbose=False)

outputs = GSEP_int['= S1']
result_file_path =  "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/280726_004.xlsx"
run_all_combinations(inputs_df, all_inputs, outputs, result_file_path,
                          model_choice='RandomForestClassifier',
                          model_params=[42, 'balanced'],
                          inputs_pca_nbr_pc=0,
                          test_size=0.2,
                          min_combo_size=1,
                          max_combo_size=None,
                          show_plot=False,
                          verbose=False)

outputs = GSEP_int['= S2']
result_file_path =  "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/280726_005.xlsx"
run_all_combinations(inputs_df, all_inputs, outputs, result_file_path,
                          model_choice='RandomForestClassifier',
                          model_params=[42, 'balanced'],
                          inputs_pca_nbr_pc=0,
                          test_size=0.2,
                          min_combo_size=1,
                          max_combo_size=None,
                          show_plot=False,
                          verbose=False)

outputs = GSEP_int['= S3']
result_file_path =  "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/280726_006.xlsx"
run_all_combinations(inputs_df, all_inputs, outputs, result_file_path,
                          model_choice='RandomForestClassifier',
                          model_params=[42, 'balanced'],
                          inputs_pca_nbr_pc=0,
                          test_size=0.2,
                          min_combo_size=1,
                          max_combo_size=None,
                          show_plot=False,
                          verbose=False)

outputs = GSEP_int['= S4']
result_file_path =  "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/280726_007.xlsx"
run_all_combinations(inputs_df, all_inputs, outputs, result_file_path,
                          model_choice='RandomForestClassifier',
                          model_params=[42, 'balanced'],
                          inputs_pca_nbr_pc=0,
                          test_size=0.2,
                          min_combo_size=1,
                          max_combo_size=None,
                          show_plot=False,
                          verbose=False)


