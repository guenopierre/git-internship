import numpy as np
#import matplotlib; matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

from GSEP_extended import build_GSEP_extended
from dataset_reading import load_srs_combine_complete

#%% functions

def histo(GSEP, srs, title, sort_by = 'Ratio (%)'):
    value_counts_gsep = GSEP.value_counts(normalize=True)*100
    value_counts_srs = srs.value_counts(normalize=True)*100
    ratio = (GSEP.value_counts()/srs.value_counts())*100

    df = pd.DataFrame({
        'SRS': value_counts_srs,
        'GSEP': value_counts_gsep, 
        'Ratio (%)': ratio
    })
    df = df.sort_values(sort_by, ascending=False)

    ax = df.plot(kind='bar', color=['grey', 'navy', 'cornflowerblue'], figsize=(10, 6))
    plt.title(title)
    plt.grid(alpha = 0.3)
    plt.show()

def histo_multi(GSEP_all, GSEP_over_S1, GSEP_over_S2, GSEP_over_S3, srs, title, sort_by='Ratio all GSEP'):
    value_counts_gsep_all = GSEP_all.value_counts(normalize=True)*100
    value_counts_gsep_over_s1 = GSEP_over_S1.value_counts(normalize=True) * 100
    value_counts_gsep_over_s2 = GSEP_over_S2.value_counts(normalize=True) * 100
    value_counts_gsep_over_s3 = GSEP_over_S3.value_counts(normalize=True) * 100

    value_counts_srs = srs.value_counts(normalize=True) * 100

    ratio_all = (GSEP_all.value_counts() / srs.value_counts()) * 100
    ratio_1 = (GSEP_over_S1.value_counts() / srs.value_counts()) * 100
    ratio_2 = (GSEP_over_S2.value_counts() / srs.value_counts()) * 100
    ratio_3 = (GSEP_over_S3.value_counts() / srs.value_counts()) * 100

    df = pd.DataFrame({
        'SRS': value_counts_srs,
        'GSEP all' : value_counts_gsep_all, 
        'GSEP >= S1': value_counts_gsep_over_s1,
        'GSEP >= S2': value_counts_gsep_over_s2,
        'GSEP >= S3': value_counts_gsep_over_s3,
        'Ratio all GSEP' : ratio_all,
        'Ratio >= S1 (%)': ratio_1,
        'Ratio >= S2 (%)': ratio_2,
        'Ratio >= S3 (%)': ratio_3
    })
    df = df.sort_values(sort_by, ascending=False)
    a =df[sort_by]
    print(a)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Graphique du haut : valeurs normalisées
    df[['SRS', 'GSEP all', 'GSEP >= S1', 'GSEP >= S2', 'GSEP >= S3']].plot(
        kind='bar', ax=ax1, color=['grey', 'navy', 'darkgoldenrod', 'darkorange', 'darkred']
    )
    ax1.set_title(title)
    ax1.set_ylabel('Répartition (%)')
    ax1.grid(alpha=0.3)

    # Graphique du bas : ratios
    df[['Ratio all GSEP', 'Ratio >= S1 (%)', 'Ratio >= S2 (%)', 'Ratio >= S3 (%)']].plot(
        kind='bar', ax=ax2, color=['navy', 'darkgoldenrod', 'darkorange', 'darkred']
    )
    ax2.set_title('Ratio [SEP event by AR per day]')
    ax2.set_ylabel('Ratio (%)')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return a 
    
#%%

date_a = pd.Timestamp('1996-01-01')
date_b = pd.Timestamp('2017-09-10')

GSEP = build_GSEP_extended()
GSEP['timestamp'] = pd.to_datetime(GSEP['timestamp'])
GSEP = GSEP[GSEP['timestamp'].between(date_a, date_b)]

GSEP_S1 = GSEP[GSEP['>= S1']==1]
GSEP_S2 = GSEP[GSEP['>= S2']==1]
GSEP_S3 = GSEP[GSEP['>= S3']==1]

srs_combine_complete = load_srs_combine_complete()
srs_combine_complete['DATETIME'] = pd.to_datetime(srs_combine_complete['DATETIME'])
srs_combine_complete = srs_combine_complete[srs_combine_complete['DATETIME'].between(date_a, date_b)]


srs_combine_complete = srs_combine_complete.rename(columns={'Z': 'AR_Mcintosh_class'})
srs_combine_complete['AR_Mcintosh_class'] = srs_combine_complete['AR_Mcintosh_class'].str.upper()

srs_combine_complete = srs_combine_complete.rename(columns={'Mag_type': 'hale_class'})
srs_combine_complete['hale_class'] = srs_combine_complete['hale_class'].str.upper()

srs_combine_complete['AR_Mcintosh_Z'] = srs_combine_complete['AR_Mcintosh_class'].str[0]
srs_combine_complete['AR_Mcintosh_p'] = srs_combine_complete['AR_Mcintosh_class'].str[1]
srs_combine_complete['AR_Mcintosh_c'] = srs_combine_complete['AR_Mcintosh_class'].str[2]

srs_combine_complete.head()

#%%

jeveux = histo_multi(GSEP['AR_Mcintosh'], GSEP_S1['AR_Mcintosh'], GSEP_S2['AR_Mcintosh'], GSEP_S3['AR_Mcintosh'], srs_combine_complete['AR_Mcintosh_class'], 'AR_Mcintosh class normalize (*100) repartition') 

histo_multi(GSEP['AR_Mcintosh_Z'], GSEP_S1['AR_Mcintosh_Z'], GSEP_S2['AR_Mcintosh_Z'], GSEP_S3['AR_Mcintosh_Z'], srs_combine_complete['AR_Mcintosh_Z'], 'AR_Mcintosh Z parameter class normalize (*100) repartition') 

histo_multi(GSEP['AR_Mcintosh_p'], GSEP_S1['AR_Mcintosh_p'], GSEP_S2['AR_Mcintosh_p'], GSEP_S3['AR_Mcintosh_p'], srs_combine_complete['AR_Mcintosh_p'], 'AR_Mcintosh p parameter class normalize (*100) repartition')

histo_multi(GSEP['AR_Mcintosh_c'], GSEP_S1['AR_Mcintosh_c'], GSEP_S2['AR_Mcintosh_c'], GSEP_S3['AR_Mcintosh_c'], srs_combine_complete['AR_Mcintosh_c'], 'AR_Mcintosh c parameter class normalize (*100) repartition') 

histo_multi(GSEP['AR_Hale'], GSEP_S1['AR_Hale'], GSEP_S2['AR_Hale'], GSEP_S3['AR_Hale'], srs_combine_complete['hale_class'], 'AR_Hale class normalize (*100) repartition')