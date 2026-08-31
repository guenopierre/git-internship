# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:22:04 2026

@author: Pierre Guéno
"""

#%% ML tailored dataset

#copy existing dataset
sep_pret_ml = sep_pret.copy()

#remove useless variables (not informative for ML)
##datetime 
sep_pret_ml = sep_pret_ml.drop(columns=["fl_start_time", "fl_peak_time", "fl_end_time", 
                                        "cme_1st_app_time", "noaa_flares_hec_id"])
##AR number 
sep_pret_ml = sep_pret_ml.drop(columns=["noaa_ar"])

#convert str values to int, and remove str
##Hale
sep_pret_ml['AR_Hale_int'] = pd.factorize(sep_pret_ml['AR_Hale'])[0] + 1 #to start at 0
sep_pret_ml = sep_pret_ml.drop(columns=["AR_Hale"])
##McIntosh
sep_pret_ml['AR_Mcintosh_int'] = pd.factorize(sep_pret_ml['AR_Mcintosh'])[0] +1 #to start at 0
sep_pret_ml = sep_pret_ml.drop(columns=["AR_Mcintosh"])