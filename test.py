import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

import sys
sys.path.append('C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/code/git-internship/')

import GSEP_extended as gsep_extended
from usefull_functions import *

#%%

GSEP = gsep_extended.GSEP_extended

GSEP['p_cme_width'] = pd.to_numeric(GSEP['p_cme_width'], errors = 'coerce')
GSEP['p_cme_speed'] = pd.to_numeric(GSEP['p_cme_width'], errors = 'coerce')
GSEP['Inst_category'] = pd.to_numeric(GSEP['Inst_category'], errors = 'coerce')
GSEP["noaa_pf10MeV"] = pd.to_numeric(GSEP["noaa_pf10MeV"], errors = 'coerce')
GSEP["noaa-sep_flag"] = pd.to_numeric(GSEP["noaa-sep_flag"], errors = 'coerce')
GSEP['radio burst 1'] = pd.to_numeric(GSEP['radio burst 1'], errors = 'coerce')
GSEP['radio burst 2'] = pd.to_numeric(GSEP['radio burst 2'], errors = 'coerce')
GSEP['AR_mag_int'] = pd.to_numeric(GSEP['AR_mag_int'], errors = 'coerce')
GSEP['AR_area'] = pd.to_numeric(GSEP['AR_area'], errors = 'coerce')
GSEP['daily_sn'] = pd.to_numeric(GSEP['daily_sn'], errors = 'coerce')

GSEP_int = GSEP.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')
GSEP_int_filtered = GSEP_int.drop(columns=['>= S1', '>= S2', '>= S3', '= S1', 
                                           '= S2', '= S3', '= S4', 'S_class',
                                           'noaa_pf10MeV', 'ppf_gt10MeV', 'gsep_fluence_gt10MeV',
                                           'fluence_gt10MeV', 'fluence_gt30MeV', 'cdaw_evn_max', 
                                           'ppf_gt30MeV', 'fluence_gt100MeV', 'ppf_gt60MeV', 
                                           'fluence_gt60MeV', 'noaa-sep_flag', 'ppf_gt100MeV', 
                                           'gsep_pf_gt10MeV', 'Flag'])

AR_params = GSEP[['AR_location', 'AR_lo', 'AR_area', 'AR_z', 'AR_ll', 'AR_nn', 'AR_mag_type', 'AR_mag_int', 'AR_z_int', 
                'AR_z_int_ranked', 'group_configuration', 'largest_spot_type', 'spots_distribution' , 'group_configuration_int', 
                'group_configuration_int_ranked', 'largest_spot_type_int', 'largest_spot_type_int_ranked', 'spots_distribution_int', 'spots_distribution_int_ranked' ]]

AR_params_int = AR_params.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')
#%% PCA

def scaler(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

GSEP_scaled = scaler(GSEP_int)

GSEP_scaled_nanfiltered = np.nan_to_num(GSEP_scaled, nan=0.0)

pca = PCA(n_components=2)
GSEP_pca = pca.fit_transform(GSEP_scaled_nanfiltered)

#%% LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(GSEP_pca, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['< S3', '>= S3'], yticklabels=['< S3', '>= S3'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#%% RandomForest
GSEP_int = GSEP.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')
X = AR_params_int
y = GSEP_int['S_class']



# 3) Split stratifié : important si les classes sont déséquilibrées
#    (ce qui est très probable ici : les événements S3/S4 sont rares)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Modèle : pas besoin de wrapper OneVsRest, RandomForest fait du multiclasse nativement
#    class_weight='balanced' aide si les classes sont déséquilibrées
model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5) Évaluation multiclasse
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1', '2', '3', '4'], 
            yticklabels=['0', '1', '2', '3', '4'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#%%
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(importance_df)

plt.figure(figsize=(6,4))
sns.barplot(data=importance_df, x='importance', y='feature', color='steelblue')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()