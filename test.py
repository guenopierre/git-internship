import pandas as pd

import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

import GSEP_extended as gsep_extended
from usefull_functions import *

#%%

GSEP = gsep_extended.GSEP_extended
GSEP['p_cme_width'] = pd.to_numeric(GSEP['p_cme_width'], errors = 'coerce')
GSEP['p_cme_speed'] = pd.to_numeric(GSEP['p_cme_width'], errors = 'coerce')
GSEP['Inst_category'] = pd.to_numeric(GSEP['Inst_category'], errors = 'coerce')
GSEP["noaa_pf10MeV"] = pd.to_numeric(GSEP["noaa_pf10MeV"], errors = 'coerce')
GSEP["noaa-sep_flag"] = pd.to_numeric(GSEP["noaa-sep_flag"], errors = 'coerce')
GSEP['radio burst 1'] = pd.to_numeric(GSEP['radio burst 1'], errors = 'coerce')
GSEP['radio burst 2'] = pd.to_numeric(GSEP['radio burst 2'], errors = 'coerce')


#########################reorganisation
GSEP_analysis = GSEP[['noaa_pf10MeV', 'gsep_pf_gt10MeV', 'ppf_gt10MeV', 'ppf_gt30MeV', 'ppf_gt60MeV', 'ppf_gt100MeV',               
             'gsep_fluence_gt10MeV', 'fluence_gt10MeV', 'fluence_gt30MeV', 'fluence_gt100MeV', 'lasco_cme_width', 'lasco_linear_speed',  
             'fl_goes_xray', ]] 

columns = GSEP_analysis.columns.tolist()

GSEP_S1 = GSEP[(GSEP["noaa_pf10MeV"] > 10) & (GSEP["noaa_pf10MeV"] <= 100)]
GSEP_S2 = GSEP[(GSEP["noaa_pf10MeV"] > 100) & (GSEP["noaa_pf10MeV"] <= 1000)]
GSEP_S3 = GSEP[(GSEP["noaa_pf10MeV"] > 1000) & (GSEP["noaa_pf10MeV"] <= 10000)]
GSEP_S4 = GSEP[GSEP["noaa_pf10MeV"] > 10000]


#%%

corr_matrix_pearson_ref1_ref2 = correlation_matrix(GSEP_analysis, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True, title='ref1 & ref2')

#%%


corr_matrix_pearson_S1 = correlation_matrix(GSEP_S1, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True, title='S1 storm')

corr_matrix_pearson_S2 = correlation_matrix(GSEP_S2, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False,   
                                          annotations=True, title='S2 storm')

corr_matrix_pearson_S3 = correlation_matrix(GSEP_S3, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True, title='S3 storm')

corr_matrix_pearson_S4 = correlation_matrix(GSEP_S4, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True, title='S4 storm')
