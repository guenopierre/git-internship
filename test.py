import pandas as pd

import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/dataset access/')
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/')

import GSEP_extended as gsep_extended
from usefull_functions import *


GSEP = gsep_extended.GSEP_extended
GSEP['p_cme_width'] = pd.to_numeric(GSEP['p_cme_width'], errors = 'coerce')
GSEP['p_cme_speed'] = pd.to_numeric(GSEP['p_cme_width'], errors = 'coerce')
GSEP['Inst_category'] = pd.to_numeric(GSEP['Inst_category'], errors = 'coerce')


colonnes = GSEP.columns.tolist()

del_params = ['sep_index',
               'cdaw_sep_id',
               'cdaw_max_time',
               'pp_index', 
               'cdaw_sep_index', 
               'timestamp', 
               'cdaw_start_time', 
               'cdaw_max_time', 
               'cdaw_evn_max', 
               'cme_id', 
               'cme_launch_time', 
               'cme_1st_app_time', 
               'fl_id', 
               'harnum',
               'fl_start_time', 
               'fl_peak_time', 
               'fl_goes_class', 
               'noaa_ar', 
               'noaa_ar_uncertain', 
               'gsep_max_time', 
               'm_type2_onset_time', 
               'dh_type2_onset_time', 
               'Comments', 
               'Flag',
               'Notes', 
               'Fe_e_p_shock_notes', 
               'gsep_notes', 
               'slice_start', 
               'slice_end',
               'Inst_category'
               'noaa-sep_flag'
               ]

columns = [col for col in colonnes if col not in del_params]

#%%

corr_matrix_spearman = correlation_matrix(GSEP, columns, 
                                          method='spearman', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True)

#%%
corr_matrix_pearson = correlation_matrix(GSEP, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True)

