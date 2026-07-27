# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:18:51 2026

@author: Pierre Guéno
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

from GSEP_extended import build_GSEP_extended
from dataset_reading import load_srs_combine_complete


#%%

GSEP = build_GSEP_extended()

GSEP = GSEP.rename(columns={'AR_z': 'mcintosh_class'})
GSEP['mcintosh_class'] = GSEP['mcintosh_class'].str.upper()

GSEP = GSEP.rename(columns={'AR_mag_type': 'hale_class'})
GSEP['hale_class'] = GSEP['hale_class'].str.upper()

GSEP['mcintosh_Z'] = GSEP['mcintosh_class'].str[0]
GSEP['mcintosh_p'] = GSEP['mcintosh_class'].str[1]
GSEP['mcintosh_c'] = GSEP['mcintosh_class'].str[2]

srs_combine_complete = load_srs_combine_complete()

srs_combine_complete = srs_combine_complete.rename(columns={'Z': 'mcintosh_class'})
srs_combine_complete['mcintosh_class'] = srs_combine_complete['mcintosh_class'].str.upper()

srs_combine_complete = srs_combine_complete.rename(columns={'Mag_type': 'hale_class'})
srs_combine_complete['hale_class'] = srs_combine_complete['hale_class'].str.upper()

srs_combine_complete['mcintosh_Z'] = srs_combine_complete['mcintosh_class'].str[0]
srs_combine_complete['mcintosh_p'] = srs_combine_complete['mcintosh_class'].str[1]
srs_combine_complete['mcintosh_c'] = srs_combine_complete['mcintosh_class'].str[2]


#%%mcintosh_class

value_counts_gsep = GSEP['mcintosh_class'].value_counts(normalize=True)
value_counts_srs = srs_combine_complete['mcintosh_class'].value_counts(normalize=True)

df = pd.DataFrame({
    'SRS': value_counts_srs,
    'GSEP': value_counts_gsep
})

df.plot(kind='bar', color=['red', 'skyblue'], figsize=(10, 6))
plt.title('McIntosh class normalize repartition')

#%%mcintosh_Z

value_counts_gsep = GSEP['mcintosh_Z'].value_counts(normalize=True)
value_counts_srs = srs_combine_complete['mcintosh_Z'].value_counts(normalize=True)

df = pd.DataFrame({
    'SRS': value_counts_srs,
    'GSEP': value_counts_gsep
})

df.plot(kind='bar', color=['red', 'skyblue'], figsize=(10, 6))
plt.title('McIntosh (Z paramater (1st)) normalize repartition')


#%%mcintosh_p

value_counts_gsep = GSEP['mcintosh_p'].value_counts(normalize=True)
value_counts_srs = srs_combine_complete['mcintosh_p'].value_counts(normalize=True)

df = pd.DataFrame({
    'SRS': value_counts_srs,
    'GSEP': value_counts_gsep
})

df.plot(kind='bar', color=['red', 'skyblue'], figsize=(10, 6))
plt.title('McIntosh (p paramater (2nd)) normalize repartition')


#%%mcintosh_c

value_counts_gsep = GSEP['mcintosh_c'].value_counts(normalize=True)
value_counts_srs = srs_combine_complete['mcintosh_c'].value_counts(normalize=True)

df = pd.DataFrame({
    'SRS': value_counts_srs,
    'GSEP': value_counts_gsep
})

df.plot(kind='bar', color=['red', 'skyblue'], figsize=(10, 6))
plt.title('McIntosh (c paramater (3rd)) normalize repartition')
