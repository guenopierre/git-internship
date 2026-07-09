#%% sys

import sys

sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/')
from usefull_functions import time_mean, convert_prefix_value

sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/dataset access/')
import dataset_reading
import sep_dictionaries

#%% reading

GSEP_extended = dataset_reading.GSEP_list

#%%

GSEP_extended['fl_goes_xray'] = GSEP_extended['fl_goes_class'].apply(convert_prefix_value)

#%% 1 - CDAW start time - CDAW max time 

_, df_clean, _ = time_mean(GSEP_extended ['cdaw_start_time'], GSEP_extended ['cdaw_max_time'], diff_max=1000)

diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended .index)

GSEP_extended ['cdaw_start_time - cdaw_max_time'] = diff_minutes

#%% 2 - CME launch time - CME 1st app time

_, df_clean, _ = time_mean(GSEP_extended ['cme_launch_time'], GSEP_extended ['cme_1st_app_time'], diff_max=1000)

diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)

GSEP_extended ['cme_launch_time - cme_1st_app_time'] = diff_minutes

#%% 3 - flare start time - flare peak time 

_, df_clean, _ = time_mean(GSEP_extended['fl_start_time'], GSEP_extended['fl_peak_time'], diff_max=1000)

diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)

GSEP_extended['fl_start_time - fl_peak_time'] = diff_minutes



