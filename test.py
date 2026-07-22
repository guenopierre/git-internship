#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

import GSEP_extended as gsep_extended
from usefull_functions import *

GSEP = gsep_extended.GSEP_extended
GSEP_int = GSEP.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')
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
all_inputs = ['daily_sn', 'AR_area', 
'AR_z_int_ranked',
'AR_mag_type_int_ranked', 
'group_configuration_int_ranked', 
'largest_spot_type_int_ranked', 
'spots_distribution_int_ranked', 
'AR_z_length_int_ranked', 
              'AR_z_penumbra_type_int_ranked'
              ]
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