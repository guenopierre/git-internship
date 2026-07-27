# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:45:48 2026

@author: Pierre Guéno
"""

import pandas as pd
import numpy as np

def load_srs_combine_complete(path_srs_combine_complete = 'C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/SWPC/SRS/srs_regions.csv'):
    return pd.read_csv(path_srs_combine_complete)


srs_combine_complete = load_srs_combine_complete()


# 1. S'assurer que la colonne DATETIME est bien au format datetime
srs_combine_complete['DATETIME'] = pd.to_datetime(srs_combine_complete['DATETIME'])

# 2. Définir la condition : année >= 2002 ET AR_number < 4000
condition = (srs_combine_complete['DATETIME'].dt.year >= 2002) & (srs_combine_complete['AR_number'] < 4000)

# 3. Créer la nouvelle colonne
# Si la condition est vraie -> ajouter 10000, sinon -> garder AR_number
srs_combine_complete['AR_number_corrected'] = np.where(condition, srs_combine_complete['AR_number'] + 10000, srs_combine_complete['AR_number'])


srs_combine_complete.to_pickle("C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/hera/srs_combine_complete_corrected.pkl")