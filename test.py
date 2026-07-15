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

colonnes = GSEP.columns.tolist()



GSEP_int = GSEP.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')



GSEP_int_filtered = GSEP_int.drop(columns=['>= S1', '>= S2', '>= S3', '= S1', 
                                           '= S2', '= S3', '= S4', 'S_class',
                                           'noaa_pf10MeV', 'ppf_gt10MeV', 'gsep_fluence_gt10MeV',
                                           'fluence_gt10MeV', 'fluence_gt30MeV', 'cdaw_evn_max', 
                                           'ppf_gt30MeV', 'fluence_gt100MeV', 'ppf_gt60MeV', 
                                           'fluence_gt60MeV', 'noaa-sep_flag', 'ppf_gt100MeV', 
                                           'gsep_pf_gt10MeV', 'Flag'])

AR_params = GSEP[['AR_location', 'AR_lo', 'AR_area', 'AR_z', 'AR_ll', 'AR_nn',  'AR_mag_type', 'AR_mag_type_int', 'AR_mag_type_int_ranked', 
                  'AR_z_int', 'group_configuration', 'largest_spot_type', 'spots_distribution', 'group_configuration_int', 
                  'largest_spot_type_int', 'spots_distribution_int', 'AR_z_magnetic_type', 'AR_z_length', 
                  'AR_z_penumbra_type', 'AR_z_distribution', 'AR_z_int_ranked', 'group_configuration_int_ranked', 
                  'largest_spot_type_int_ranked', 'spots_distribution_int_ranked', 'AR_z_magnetic_type_int_ranked', 
                  'AR_z_length_int_ranked', 'AR_z_penumbra_type_int_ranked', 'AR_z_distribution_int_ranked']]

AR_params_int = AR_params.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')

columns = AR_params_int.columns.tolist()
columns.append('noaa_pf10MeV')

#%% correlation matrix 

corr_matrix_pearson = correlation_matrix(GSEP, columns, 
                                          method='pearson', plot = True, 
                                          interactive=True, cr=False, 
                                          annotations=True, title='ARs parameters')
    

#%% PCA



pca1, GSEP_pca1 = run_pca(AR_params_int, correlation_circle=True)





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


AR_params = GSEP[['AR_location', 'AR_lo', 'AR_area', 'AR_z', 'AR_ll', 'AR_nn',  'AR_mag_type', 'AR_mag_type_int', 'AR_mag_type_int_ranked', 
                  'AR_z_int', 'group_configuration', 'largest_spot_type', 'spots_distribution', 'group_configuration_int', 
                  'largest_spot_type_int', 'spots_distribution_int', 'AR_z_magnetic_type', 'AR_z_length', 
                  'AR_z_penumbra_type', 'AR_z_distribution', 'AR_z_int_ranked', 'group_configuration_int_ranked', 
                  'largest_spot_type_int_ranked', 'spots_distribution_int_ranked', 'AR_z_magnetic_type_int_ranked', 
                  'AR_z_length_int_ranked', 'AR_z_penumbra_type_int_ranked', 'AR_z_distribution_int_ranked']]

AR_params_int = AR_params.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')


GSEP_int = GSEP.select_dtypes(include=['int', 'float']).dropna(axis=1, how='all')
X = AR_params_int  # à ajuster 
y = GSEP_int['S_class'] # à ajuster


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1', '2', '3', '4'], 
            yticklabels=['0', '1', '2', '3', '4'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

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