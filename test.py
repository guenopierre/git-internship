import pandas as pd

import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

import GSEP_extended as gsep_extended
from usefull_functions import *

GSEP = gsep_extended.GSEP_extended
GSEP['p_cme_width'] = pd.to_numeric(GSEP['p_cme_width'], errors = 'coerce')
GSEP['p_cme_speed'] = pd.to_numeric(GSEP['p_cme_width'], errors = 'coerce')
GSEP['Inst_category'] = pd.to_numeric(GSEP['Inst_category'], errors = 'coerce')
GSEP["noaa_pf10MeV"] = pd.to_numeric(GSEP["noaa_pf10MeV"], errors = 'coerce')
GSEP["noaa-sep_flag"] = pd.to_numeric(GSEP["noaa-sep_flag"], errors = 'coerce')



GSEP_S1 = GSEP[(GSEP["noaa_pf10MeV"] > 10) & (GSEP["noaa_pf10MeV"] <= 100)]
GSEP_S2 = GSEP[(GSEP["noaa_pf10MeV"] > 100) & (GSEP["noaa_pf10MeV"] <= 1000)]
GSEP_S3 = GSEP[(GSEP["noaa_pf10MeV"] > 1000) & (GSEP["noaa_pf10MeV"] <= 10000)]
GSEP_S4 = GSEP[GSEP["noaa_pf10MeV"] > 10000]

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
               'Inst_category', 
               'harpnum', 
               'noaa-sep_flag'
               ]

columns = [col for col in colonnes if col not in del_params]


#%%


corr_matrix_pearson_S1 = correlation_matrix(GSEP_S1, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True)

corr_matrix_pearson_S2 = correlation_matrix(GSEP_S2, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False,   
                                          annotations=True)

corr_matrix_pearson_S3 = correlation_matrix(GSEP_S3, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True)

corr_matrix_pearson_S4 = correlation_matrix(GSEP_S4, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True)
