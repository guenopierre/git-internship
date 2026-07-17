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


inputs_df = GSEP_int
all_inputs = ['daily_sn', 'AR_area', 
              'AR_z_int','AR_z_int_ranked',
              'AR_mag_type_int', 'AR_mag_type_int_ranked', 
              'group_configuration_int', 'group_configuration_int_ranked', 
              'largest_spot_type_int', 'largest_spot_type_int_ranked', 
              'spots_distribution_int', 'spots_distribution_int_ranked', 
              'AR_z_length_int', 'AR_z_length_int_ranked', 
              'AR_z_penumbra_type_int_ranked', 'AR_z_penumbra_type_int'
              ]
outputs = GSEP_int['>= S1']
result_file_path =  "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/170726_008.xlsx"


run_all_combinations(inputs_df, all_inputs, outputs, result_file_path,
                          model_choice='RandomForestClassifier',
                          model_params=[42, 'balanced'],
                          inputs_pca_nbr_pc=0,
                          test_size=0.2,
                          min_combo_size=1,
                          max_combo_size=None,
                          show_plot=False,
                          verbose=False)