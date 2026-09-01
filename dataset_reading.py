"""
Reading all the datasets from their associated path on my laptop
/!\ Doesn't work for HERA
"""

#%% Librairies 
import pickle
import warnings
import pandas as pd
import numpy as np

#%% Warning removal
warnings.filterwarnings(
    "ignore", category=DeprecationWarning,
)   #remove all depreciation warning

#%% SEP PRET

def load_sep_pret(version = 4):
    if version == 4:
        with open("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v4.pkl", 'rb') as file:
            return pickle.load(file)
    elif version == 3:
        with open("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v3.pkl", 'rb') as file:
            return pickle.load(file)
    elif version ==2:
        with open("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v2.pkl", 'rb') as file:
            return pickle.load(file)
    
    
def load_sep_pret_reduced(version = 4):
    if version == 4:
        with open("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v4_reduced.pkl", 'rb') as file:
            return pickle.load(file)
    elif version == 3:
        with open("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v3_reduced.pkl", 'rb') as file:
            return pickle.load(file)
    elif version == 2:
        with open("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v2_reduced.pkl", 'rb') as file:
            return pickle.load(file)
    
    
def load_sep_pret_train_test(version=4):
    if version == 4:
        with (
            open("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v4_train.pkl", 'rb') as train,
            open("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v4_sample_50.pkl", 'rb') as train_sample_50,
            open("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_v4_test.pkl", 'rb') as test
        ):
            return pickle.load(train), pickle.load(train_sample_50), pickle.load(test)
#%% NOAA FLARES
def load_noaa_flares(path_noaa_flares = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/noaa_flares_mar76-jan25.pkl'):
    """
    Source: Hera (internal)

    Parameters
    ----------
    path_noaa_flares : str, optional
        The default is 'C:/Users/pierr/OneDrive - IPsSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/noaa_flares_mar76-jan25.pkl'.

    Returns
    -------
    dataframe
        noaa_flares

    """
    with open(path_noaa_flares, 'rb') as file:
        return pickle.load(file)

def load_noaa_flares_extended(path_noaa_flares_extended = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/noaa_flares_extended.pkl'):
    with open(path_noaa_flares_extended, 'rb') as file:
        return pickle.load(file)
    
def load_noaa_flares_c1(path = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/noaa_flares_c1_threshold.pkl" ):
    with open(path, 'rb') as file:
        return pickle.load(file)
#%% FORMAL
def load_formal_sep(path_formal_sep = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/formal-sep_all_data.pkl'):
    with open(path_formal_sep, 'rb') as file:
        return pickle.load(file)
#%% CDAW (SEP Extended & CME)
def load_cdaw_sepe_extented(path_cdaw_sepe_extented = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/cdaw_sepe_list_extended.pkl'):
    with open(path_cdaw_sepe_extented, 'rb') as file:
        return pickle.load(file)

def load_cdaw_cme(path_cdaw_cme = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/cme_cdaw_19960111-20240930.pkl'):
    with open(path_cdaw_cme, 'rb') as file:
        return pickle.load(file)
    
def load_cdaw_reduced_online_version(path_cdaw = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/CDAW.csv"):
    return pd.read_csv(path_cdaw, sep= ";")
#%% SRS COMBINE
def load_srs_combine(path_srs_combine = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/SRS_Combine_1996_2024.pkl'):
    with open(path_srs_combine, 'rb') as file:
        return pickle.load(file)

def load_srs_combine_complete(path_srs_combine_complete = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/SWPC/SRS/srs_regions.csv'):
    return pd.read_csv(path_srs_combine_complete)

def load_srs_combine_complete_corrected(path_srs_combine_complete_corrected = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/srs_combine_complete_corrected.pkl"):
    with open(path_srs_combine_complete_corrected, 'rb') as file:
        return pickle.load(file)
#%% GSEP
def load_GSEP(path_GSEP = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/GSEP"):
    return pd.read_csv(path_GSEP + "/GSEP_list.csv")

def load_GSEP_extended():
    import GSEP_extended
    return GSEP_extended.build_GSEP_extended()

def load_GSEP_int_extended():
    import GSEP_extended
    return GSEP_extended.build_GSEP_int_extended()
#%% MEMPSEP
def load_mempsep(path_mempsep = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/1998_2013_MEMSEP_dataset.csv'):
    return pd.read_csv(path_mempsep)
#%% SEPEM
def load_sepem_FADO_FAIO(path_sepem_FADO_FAIO = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEPEM_RDS_v3_2/RDS3.2_FADO_FAIO.csv'):
    return pd.read_csv(path_sepem_FADO_FAIO)

def load_sepem_FPDO_FPIO(path_sepem_FPDO_FPIO = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEPEM_RDS_v3_2/RDS3.2_FADO_FAIO.csv'):
    return pd.read_csv(path_sepem_FPDO_FPIO)

#%% Laurenza - ESPERTA
def load_laurenza_esperta(path_laurenza = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/ESPERTA Laurenza 2009 Solar Energetic Particle Flare Data Table.csv"):
    return pd.read_csv(path_laurenza, sep=';')

#%% Sunspot Number (SN)
def load_SN_d_tot_V2(path_SN_d_tot_V2 = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/sunspot/SN_d_tot_V2.0.csv"):
    return pd.read_csv(path_SN_d_tot_V2, delimiter=";", header=None, 
                       names=['year', 'month', 'day', 'date_decimal', 'daily_total_sn',
                              'daily_std_variation', 'nb_observations', 'definitive/provisional'])

#%% PRISM 
def load_PRISM_processed_CDAW_CME(path_sep_prism_processed = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP-PRISM/SEP-PRISM/Processed Data"):
    return pd.read_csv(path_sep_prism_processed + "/CDAW_CME.csv")

def load_PRISM_processed_CDAWDONKI_CME(path_sep_prism_processed = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP-PRISM/SEP-PRISM/Processed Data"):
    return pd.read_csv(path_sep_prism_processed + "/CDAWDONKI_CME.csv")

def load_PRISM_processed_Flare(path_sep_prism_processed = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP-PRISM/SEP-PRISM/Processed Data"):
    return pd.read_csv(path_sep_prism_processed + "/Flare.csv")

def load_PRISM_processed_GOESHAPI_ProtonFlux(path_sep_prism_processed = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP-PRISM/SEP-PRISM/Processed Data"):
    return pd.read_csv(path_sep_prism_processed + "/GOESHAPI_ProtonFlux.csv")

def load_PRISM_processed_Merged_XRays(path_sep_prism_processed = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP-PRISM/SEP-PRISM/Processed Data"):
    return pd.read_csv(path_sep_prism_processed + "/Merged_XRays.csv")

def load_PRISM_processed_SMHARP(path_sep_prism_processed = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP-PRISM/SEP-PRISM/Processed Data"):
    return pd.read_csv(path_sep_prism_processed + "/SMHARP.csv")

def load_PRISM_analyzed_rolling_combinded_seq_1hours(path_sep_prism_analyzed = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP-PRISM/SEP-PRISM/Analyzed Data"):
    return pd.read_csv(path_sep_prism_analyzed + "/rolling_combinded_seq_1hours.csv")

def load_PRISM_analyzed_rolling_combinded_seq_24hours(path_sep_prism_analyzed = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP-PRISM/SEP-PRISM/Analyzed Data"):
    return pd.read_csv(path_sep_prism_analyzed + "/rolling_combinded_seq_24hours.csv")
#%% SolARED
def load_solared(path_solARed = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SolARED/active_regions_export_2026-07-02.csv'):
    return pd.read_csv(path_solARed, delimiter=',')

#%% %DONKI

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



#%% %HAPI ISWA API

# def hapi_csv(url = "https://iswa.ccmc.gsfc.nasa.gov/hapi/data?id=goesp_mag_p1m&time.min=2018-04-25T00:00:00.0Z&time.max=2018-04-26T00:00:00.0Z"):
#     response = requests.get(url)
#     with open("downloaded_data.csv", "wb") as file:
#         file.write(response.content)
#     df = pd.read_csv("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/test HAPI/downloaded_data.csv")
#     return df

# print("HAPI function imported")


#%% %IMAP (currently unauthorized)

# import ialirt_data_access
# results = ialirt_data_access.log_query(year="2026", doy="150", instance="1")
# fichiers = list(results.values())[0]
# ialirt_data_access.download(
#     filename=fichiers[3],  # premier fichier
#     filetype="logs"
# )



