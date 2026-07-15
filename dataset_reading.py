import numpy as np
import pickle
import warnings
import pandas as pd
import requests
from datetime import datetime, timedelta
import seaborn

warnings.filterwarnings(
    "ignore", category=DeprecationWarning,
)   #remove all depreciation warning


#%% HERA (NOAA, CDAW, Formal SEP, SRS)

path_noaa_flares = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/noaa_flares_mar76-jan25.pkl'
path_formal_sep = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/formal-sep_all_data.pkl'
path_cdaw_cme = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/cme_cdaw_19960111-20240930.pkl'
path_cdaw_sepe_extented = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/cdaw_sepe_list_extended.pkl'
path_srs_combine = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/SRS_Combine_1996_2024.pkl'

# Open the file in binary mode and load the data
with open(path_noaa_flares, 'rb') as file:
    noaa_flares = pickle.load(file)
    
with open(path_formal_sep, 'rb') as file:
    formal_sep = pickle.load(file)
    
with open(path_cdaw_sepe_extented, 'rb') as file:
    cdaw_sepe_extented = pickle.load(file)
    
with open(path_srs_combine, 'rb') as file:
    srs_combine = pickle.load(file)


    
with open(path_cdaw_cme, 'rb') as file:
    cdaw_cme = pickle.load(file)
    
 
del file, path_noaa_flares, path_formal_sep, path_cdaw_cme, path_cdaw_sepe_extented, path_srs_combine

print("hera datasets imported")


#%%

path_srs_combine_complete = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/SWPC/SRS/srs_regions.csv'
srs_combine_complete = pd.read_csv(path_srs_combine_complete)

print("srs_combine_complete imported")

#%% GSEP 

path_GSEP = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/GSEP"

GSEP_list = pd.read_csv(path_GSEP + "/GSEP_list.csv")

del path_GSEP

print("GSEP imported")

#%% MEMPSEP

path_mempsep = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/1998_2013_MEMSEP_dataset.csv'

mempsep = pd.read_csv(path_mempsep)

del path_mempsep


#%% SEPEM

# path_sepem_FADO_FAIO = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEPEM_RDS_v3_2/RDS3.2_FADO_FAIO.csv'
# path_sepem_FPDO_FPIO = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEPEM_RDS_v3_2/RDS3.2_FPDO_FPIO.csv'

# sepem_FADO_FAIO = pd.read_csv(path_sepem_FADO_FAIO)
# sepem_FPDO_FPIO = pd.read_csv(path_sepem_FPDO_FPIO)

# del path_sepem_FADO_FAIO, path_sepem_FPDO_FPIO

# print("SEPEM imported")
#%% CDAW (reduced -- online)

# cdaw = pd.read_csv("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/CDAW.csv", sep= ";")

# print("CDAW imported")
#%% DONKI

# def get_DONKI_sep_events(start_date=None, end_date=None, catalog="ALL", version="Latest"):
#     """
#     Retrieves SEP events from NASA’s DONKI API 
#     https://ccmc.gsfc.nasa.gov/tools/DONKI/#donki-webservice-calls-api

#     Settings:
#     - start_date: start date (format 'YYYY-MM-DD')
#     - end_date: end date (format 'YYYY-MM-DD')
#     - catalog: catalog type (ALL or M2M_CATALOG)
#     - version: data version (ALL or Latest)

#     Returns:
#     - A Python dictionary containing the JSON data
#     """
    

#     # Use default dates if not specified
#     if start_date is None:
#         start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
#     if end_date is None:
#         end_date = datetime.utcnow().strftime('%Y-%m-%d')

#     # Build API URL
#     base_url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/SEP"
#     params = {
#         'startDate': start_date,
#         'endDate': end_date,
#         'catalog': catalog,
#         'version': version
#     }

#     try:
#         # Make the HTTP request
#         response = requests.get(base_url, params=params)
#         response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP

#         # Parse the JSON response
#         data = response.json()
#         return data

#     except requests.exceptions.RequestException as e:
#         print(f"Erreur lors de la requête API: {e}")
#         return None

# donki_default = get_DONKI_sep_events(start_date='2000-01-01', end_date='2015-01-01')


# print("Donki imported")
#%% Laurenza

laurenza = pd.read_csv("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/ESPERTA Laurenza 2009 Solar Energetic Particle Flare Data Table.csv", sep=';')
print("Laurenza imported")


#%% HAPI ISWA API

# def hapi_csv(url = "https://iswa.ccmc.gsfc.nasa.gov/hapi/data?id=goesp_mag_p1m&time.min=2018-04-25T00:00:00.0Z&time.max=2018-04-26T00:00:00.0Z"):
#     response = requests.get(url)
#     with open("downloaded_data.csv", "wb") as file:
#         file.write(response.content)
#     df = pd.read_csv("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/test HAPI/downloaded_data.csv")
#     return df

# print("HAPI function imported")


#%% IMAP (currently unauthorized)

# import ialirt_data_access
# results = ialirt_data_access.log_query(year="2026", doy="150", instance="1")
# fichiers = list(results.values())[0]
# ialirt_data_access.download(
#     filename=fichiers[3],  # premier fichier
#     filetype="logs"
# )

#%%

path_solARed = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SolARED/active_regions_export_2026-07-02.csv'

solARed = pd.read_csv(path_solARed, delimiter=',')

#%%

