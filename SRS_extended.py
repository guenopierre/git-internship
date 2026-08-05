"""
add the corrected AR number (+10 000 since 2002)
convert the str location into int lat & long columns
"""

#%% Librairies
import pandas as pd
import numpy as np
import re

#%% Datasets 
import dataset_reading
srs_combine_complete = dataset_reading.load_srs_combine_complete()

#%% AR number correction 
srs_combine_complete['DATETIME'] = pd.to_datetime(srs_combine_complete['DATETIME'])
condition = (srs_combine_complete['DATETIME'].dt.year >= 2002) & (srs_combine_complete['AR_number'] < 4000)
srs_combine_complete['AR_number_corrected'] = np.where(condition, srs_combine_complete['AR_number'] + 10000, srs_combine_complete['AR_number'])

#%% Location Type Conversion
def parse(s):
    if not isinstance(s, str):
        return pd.Series([None, None])
    
    # Latitude
    lat_match = re.search(lat_pattern, s)
    if lat_match:
        lat_sign, lat_val = lat_match.groups()
        lat = float(lat_val) * (1 if lat_sign == 'N' else -1)
    else:
        lat = None
    
    # Longitude
    long_match = re.search(long_pattern, s)
    if long_match:
        long_sign, long_val = long_match.groups()
        long = float(long_val) * (1 if long_sign == 'W' else -1)
    else:
        long = None
    
    return pd.Series([lat, long])

lat_pattern = r'([NS])(\d+\.?\d*)'
long_pattern = r'([EW])(\d+\.?\d*)'

srs_combine_complete[['AR_lat', 'AR_long']] = srs_combine_complete['Location'].apply(parse)

#%% Export
srs_combine_complete.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/srs_combine_complete_corrected.pkl")