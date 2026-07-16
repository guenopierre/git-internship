#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
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

X = GSEP[['fl_goes_x_ray', 'fl_lon', 'lasco_linear_speed', 'AR_area', 'daily_sn']]
y = GSEP['>= S3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- METRICS
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn); print(f"Accuracy (Acc): {accuracy:.4f}")
pod = tp / (tp + fn); print(f"Probability of Detection (POD): {pod:.4f}")
far = fp / (fp + tp); print(f"False Alarm Ratio (FAR): {far:.4f}")
prec = tp / (tp + fp); print(f"Precision (Prec): {prec:.4f}")
f1 = 2 * (prec * pod) / (prec + pod); print(f"F1 Score: {f1:.4f}")
tss = pod - far; print(f"True Skill Statistic (TSS): {tss:.4f}")
hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)); print(f"Heidke Skill Score (HSS): {hss:.4f}")


# --- CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], 
            yticklabels=['0', '1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# --- IMPORTANCES

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

#%%

GSEP_S2_3_4 = GSEP[GSEP['>= S2'] != 0]

X = GSEP_S2_3_4[['lasco_cme_width', 'fl_lon', 'fl_goes_xray', 'AR_z_int_ranked']]
y = GSEP_S2_3_4['>= S3']

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['S2', 'S3 or S4'], 
            yticklabels=['S2', 'S3 or S4'])
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