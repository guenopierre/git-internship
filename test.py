#%% librairies

import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import LogNorm

sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/')
from usefull_functions import *

sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/dataset access/')
import dataset_reading
import sep_dictionaries
import GSEP_extended as gsep_extended


#%%




GSEP_list = dataset_reading.GSEP_list.dropna(subset=['cdaw_evn_max', 'lasco_cme_width', 'lasco_linear_speed', 'fl_rise_time', 
                                                     'fl_lon', 'fl_lat','fl_goes_class', 'noaa_pf10MeV', 'fluence_gt10MeV'])

GSEP_list['fl_goes_xray'] = GSEP_list['fl_goes_class'].apply(convert_prefix_value)

scatter_parameters(GSEP_list['fl_goes_xray'], GSEP_list['noaa_pf10MeV'], 
                   name_p1= 'fl_goes_xray', name_p2='noaa_pf10MeV')


#%%
import sys

GSEP = gsep_extended.GSEP_extended

corr_matrix_spearman = correlation_matrix(GSEP, 
                                        ['cdaw_evn_max', 'lasco_cme_width', 'lasco_linear_speed',
                                        'fl_rise_time', 'fl_lon', 'fl_lat', 'fl_goes_xray', 
                                        'noaa_pf10MeV', 'fluence_gt10MeV', 
                                        'cdaw_start_time - cdaw_max_time', 
                                        'cme_launch_time - cme_1st_app_time', 
                                        'fl_start_time - fl_peak_time'], 
                                          method='spearman', plot = True)

corr_matrix_pearson = correlation_matrix(GSEP, 
                                        ['cdaw_evn_max', 'lasco_cme_width', 'lasco_linear_speed',
                                        'fl_rise_time', 'fl_lon', 'fl_lat', 'fl_goes_xray', 
                                        'noaa_pf10MeV', 'fluence_gt10MeV', 
                                        'cdaw_start_time - cdaw_max_time', 
                                        'cme_launch_time - cme_1st_app_time', 
                                        'fl_start_time - fl_peak_time'], 
                                        method='pearson', plot = True)

#%%

diff_pearson_spearman = abs(corr_matrix_spearman - corr_matrix_pearson)


