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

#%%

_, df_clean, _ = time_mean(GSEP_extended ['cdaw_start_time'], GSEP_extended ['timestamp'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended .index)
GSEP_extended ['cdaw_start_time ref'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended ['cme_launch_time'], GSEP_extended ['timestamp'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended ['cme_launch_time ref'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended['fl_start_time'], GSEP_extended['timestamp'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended['fl_start_time ref'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended['fl_peak_time'], GSEP_extended['timestamp'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended['fl_peak_time ref'] = diff_minutes

_, df_clean, _ = time_mean(GSEP_extended['slice_start'], GSEP_extended['slice_end'], diff_max=1000)
diff_minutes = (df_clean['difference'].dt.total_seconds() / 60).reindex(GSEP_extended.index)
GSEP_extended['slice range'] = diff_minutes

#%%

GSEP_extended['CME flag'] = GSEP_extended['cme_id'].notna().astype(int)

GSEP_extended['flare flag'] = GSEP_extended['fl_id'].notna().astype(int)

GSEP_extended['CME + flare flag'] = (GSEP_extended['cme_id'].notna() & GSEP_extended['fl_id'].notna()).astype(int)
