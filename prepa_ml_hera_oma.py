import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split # /!\ find an alternative bc hera numpy doesn't accept
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

import itertools
from pathlib import Path
#%%

path_sep_pret = "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/dataset/SEP PRET/sep_pret_ml_001.pkl"

with open(path_sep_pret, 'rb') as file:
    sep_pret_ml = pickle.load(file)

del path_sep_pret, file
    
columns = sep_pret_ml.columns.tolist()

#%%
 
def run_ml_binary_classification_sklearn(
    inputs: pd.DataFrame,
    output: pd.DataFrame,
    test_size: float = 0.2,
    model_type: str = "random_forest",
    random_state: int = 42,
    verbose: bool = True,
    rf_params: dict = None,
):
    """
    Train and evaluate a binary classification model.
 
    Parameters
    ----------
    inputs : pd.DataFrame
        Feature matrix (X). All columns should be numeric.
    output : pd.DataFrame or pd.Series
        Target containing a binary flag (0 or 1). If a DataFrame is given,
        it must contain exactly one column.
    test_size : float, default 0.2
        Proportion of the dataset to allocate to the test split.
    model_type : str, default "random_forest"
        Which model to train. Currently supported:
            - "random_forest" : sklearn.ensemble.RandomForestClassifier
        (Additional model types, e.g. a PyTorch neural network with CUDA
        support, can be added later as extra branches.)
    random_state : int, default 42
        Random seed used for the train/test split and the Random Forest.
    verbose : bool, default True
        If True, prints the train/test class balance check, the final
        metrics, and the feature importances (when available).
    rf_params : dict, optional
        Extra keyword arguments forwarded to RandomForestClassifier.
 
    Returns
    -------
    results : dict
        {
            "model": trained model object,
            "model_type": str,
            "confusion_matrix": np.ndarray (2x2) -> [[TN, FP], [FN, TP]],
            "accuracy": float,
            "pod": float,   # Probability Of Detection (a.k.a. Recall/Sensitivity)
            "far": float,   # False Alarm Ratio
            "precision": float,
            "f1_score": float,
            "tss": float,   # True Skill Statistic (Hanssen-Kuipers / Peirce Skill Score)
            "hss": float,   # Heidke Skill Score
            "train_class1_ratio": float,
            "test_class1_ratio": float,
            "feature_importances": pd.Series or None,
                # Index = feature name, value = importance, sorted descending.
                # Populated for models exposing `feature_importances_`
                # (e.g. RandomForestClassifier) or `coef_` (linear models).
                # None if the model type does not expose either.
        }
    """
 
    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    if isinstance(output, pd.DataFrame):
        if output.shape[1] != 1:
            raise ValueError("`output` must be a DataFrame with exactly one column (the binary flag).")
        y = output.iloc[:, 0].to_numpy()
    elif isinstance(output, pd.Series):
        y = output.to_numpy()
    else:
        raise TypeError("`output` must be a pandas DataFrame or Series.")
 
    if not isinstance(inputs, pd.DataFrame):
        raise TypeError("`inputs` must be a pandas DataFrame.")
 
    unique_vals = set(np.unique(y))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"`output` must only contain 0/1 flags, got values: {unique_vals}")
 
    if model_type not in ("random_forest",):
        raise ValueError('`model_type` must be "random_forest" (only option available in this sklearn-only version).')
 
    if not (0.0 < test_size < 1.0):
        raise ValueError("`test_size` must be between 0 and 1.")
 
    non_numeric_cols = inputs.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            f"`inputs` contains non-numeric columns {non_numeric_cols}. "
            "Please encode categorical variables before calling this function."
        )
 
    feature_names = inputs.columns.tolist()
    X = inputs.to_numpy(dtype=np.float64)
    y = y.astype(np.int64)
 
    # ------------------------------------------------------------------
    # 1. Train / test split, stratified to preserve the 0/1 proportion
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # guarantees the same class 0/1 proportion in train and test
    )
 
    train_ratio = float(np.mean(y_train))
    test_ratio = float(np.mean(y_test))
 
    if verbose:
        print("=" * 60)
        print("TRAIN / TEST SPLIT SUMMARY")
        print("=" * 60)
        print(f"Train set size                     : {len(y_train)} samples")
        print(f"Test set size                       : {len(y_test)} samples")
        print(f"Proportion of class 1 in TRAIN set  : {train_ratio:.4f}")
        print(f"Proportion of class 1 in TEST set   : {test_ratio:.4f}")
        print(f"Absolute difference (train - test)  : {abs(train_ratio - test_ratio):.4f}")
        print("=" * 60)
 
    # ------------------------------------------------------------------
    # 2. Model training
    # ------------------------------------------------------------------
    rf_params = dict(rf_params) if rf_params else {}
    rf_params.setdefault("random_state", random_state)
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    # ------------------------------------------------------------------
    # 2bis. Feature importance (only computed if the model exposes it)
    # ------------------------------------------------------------------
    feature_importances = None
 
    if hasattr(model, "feature_importances_"):
        # Tree-based models (Random Forest, Gradient Boosting, ...)
        importances = model.feature_importances_
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        # Linear models (Logistic Regression, ...): use |coefficient| as an importance proxy
        importances = np.abs(np.ravel(model.coef_))
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
 
    # ------------------------------------------------------------------
    # 3. Metrics
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
 
    # POD (Probability Of Detection) = Recall = Sensitivity = TP / (TP + FN)
    pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
 
    # FAR (False Alarm Ratio) = fraction of positive PREDICTIONS that were wrong
    # NOTE: this is different from the "False Alarm Rate" (a.k.a. POFD), which is FP / (FP + TN)
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
 
    # POFD (Probability Of False Detection), needed to compute TSS
    pofd = fp / (fp + tn) if (fp + tn) > 0 else np.nan
 
    # TSS (True Skill Statistic, a.k.a. Hanssen-Kuipers / Peirce Skill Score)
    tss = pod - pofd if not (np.isnan(pod) or np.isnan(pofd)) else np.nan
 
    # HSS (Heidke Skill Score)
    hss_denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = (2 * (tp * tn - fp * fn) / hss_denom) if hss_denom > 0 else np.nan
 
    results = {
        "model": model,
        "model_type": model_type,
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "pod": pod,
        "far": far,
        "precision": precision,
        "f1_score": f1,
        "tss": tss,
        "hss": hss,
        "train_class1_ratio": train_ratio,
        "test_class1_ratio": test_ratio,
        "feature_importances": feature_importances,  # pandas Series, sorted desc, or None
    }
 
    if verbose:
        print("\n" + "=" * 60)
        print(f"RESULTS - model_type = '{model_type}'")
        print("=" * 60)
        print("Confusion Matrix:")
        print("                    Predicted 0     Predicted 1")
        print(f"Actual 0        |      {tn:>6}          {fp:>6}")
        print(f"Actual 1        |      {fn:>6}          {tp:>6}")
        print("-" * 60)
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"POD        : {pod:.4f}")
        print(f"FAR        : {far:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"TSS        : {tss:.4f}")
        print(f"HSS        : {hss:.4f}")
        print("=" * 60)
 
        if feature_importances is not None:
            print("\n" + "=" * 60)
            print("FEATURE IMPORTANCE")
            print("=" * 60)
            for feat_name, importance in feature_importances.items():
                print(f"{feat_name:<30}: {importance:.4f}")
            print("=" * 60)
        else:
            print(f"\nFeature importance is not available for model_type = '{model_type}'.")
 
    return results

def loop_test_params_combinations(
    inputs: pd.DataFrame,
    output: pd.DataFrame,
    excel_path: str,
    test_size: float = 0.2,
    model_type: str = "random_forest",
    random_state: int = 42,
    verbose: bool = False,
    rf_params: dict = None,
) -> pd.DataFrame:
    """
    Teste toutes les combinaisons possibles de colonnes du DataFrame `inputs`,
    exécute l'entraînement/évaluation du modèle et enregistre les résultats dans
    un fichier Excel.

    Parameters
    ----------
    inputs : pd.DataFrame
        DataFrame contenant toutes les features disponibles.
    output : pd.DataFrame ou pd.Series
        Variable cible binaire (0 ou 1).
    excel_path : str ou Path (obligatoire)
        Chemin du fichier Excel (créé ou mis à jour).
    test_size : float, default 0.2
        Proportion d'échantillons attribués au jeu de test.
    model_type : str, default "random_forest"
        Type de modèle à utiliser.
    random_state : int, default 42
        Garantit que le split Train/Test est rigoureusement identique
        pour toutes les combinaisons.
    verbose : bool, default False
        Contrôle le niveau d'affichage de la fonction `run_ml_binary_classification_sklearn`.
    rf_params : dict, optional
        Paramètres supplémentaires pour RandomForestClassifier.

    Returns
    -------
    pd.DataFrame
        Tableau récapitulatif avec les paramètres/features en lignes et les
        combinaisons en colonnes.
    """
    feature_names = inputs.columns.tolist()
    num_features = len(feature_names)
    
    if num_features == 0:
        raise ValueError("Le DataFrame `inputs` ne contient aucune colonne.")

    # Liste des métriques à enregistrer (extraites du dictionnaire de résultats)
    metric_keys = [
        "accuracy",
        "pod",
        "far",
        "precision",
        "f1_score",
        "tss",
        "hss",
        "train_class1_ratio",
        "test_class1_ratio",
    ]

    # Première colonne (index) : Liste de toutes les features puis des métriques
    index_labels = feature_names + metric_keys
    results_dict = {}

    # 1. Génération de toutes les combinaisons de 1 à N features
    total_combinations = (2 ** num_features) - 1
    if verbose:
        print(f"Début des tests : {total_combinations} combinaison(s) à évaluer...")

    comb_counter = 1
    for r in range(1, num_features + 1):
        for combo in itertools.combinations(feature_names, r):
            combo_cols = list(combo)
            combo_name = f"Comb_{comb_counter}"
            print(f"Combinaison {comb_counter} over {total_combinations}")
            # Sélection du sous-ensemble de variables
            X_sub = inputs[combo_cols]

            # 2. Exécution du modèle
            # L'utilisation constante de `random_state` garantit que le train_test_split
            # conserve EXACTEMENT les mêmes index de lignes à chaque itération.
            res = run_ml_binary_classification_sklearn(
                inputs=X_sub,
                output=output,
                test_size=test_size,
                model_type=model_type,
                random_state=random_state,
                verbose=verbose,
                rf_params=rf_params,
            )

            # 3. Remplissage des cellules pour la combinaison courante
            column_data = {}
            importances = res.get("feature_importances")

            # Remplissage des features (Importance si présente dans la combinaison, NaN sinon)
            for feat in feature_names:
                if feat in combo_cols and importances is not None and feat in importances:
                    column_data[feat] = importances[feat]
                else:
                    column_data[feat] = np.nan

            # Remplissage des métriques
            for metric in metric_keys:
                column_data[metric] = res.get(metric, np.nan)

            results_dict[combo_name] = column_data
            comb_counter += 1

    # Construction du DataFrame global
    df_results = pd.DataFrame(results_dict, index=index_labels)
    df_results.index.name = "Paramètres & Métriques"

    # 4. Sauvegarde ou mise à jour du fichier Excel
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)  # Crée les dossiers si besoin

    if excel_path.exists():
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_results.to_excel(writer, sheet_name="Resultats_Combinaisons")
    else:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_results.to_excel(writer, sheet_name="Resultats_Combinaisons")

    if verbose:
        print(f"Résultats sauvegardés avec succès dans : {excel_path}")

    return df_results

#%%
inputs_df = sep_pret_ml[['fl_rising_time', 'fl_total_time', 'fl_goes_xray', 'fl_lon', 
                         'noaa_flares_AR_flares_count_last24h', 'noaa_flares_AR_xray_max_last24h', 
                         'AR_Area', 'cme_rising_time', 'lasco_linear_speed', 
                         'lasco_cme_width', 'daily_sn']]
output_df = sep_pret_ml[['GSEP flag']]
res_rf = run_ml_binary_classification_sklearn(inputs_df, output_df)

#%%

#/!\ takes a lot of time 
# for 2047 combinations -> more than an hour
start = time.perf_counter()
loop_test_params_combinations(
    inputs_df,
    output_df,
    "C:/Users/pierr/OneDrive - IPSA/Documents/IPSA/Aero 4/Stage A4/BIRA IASB Bruxelles/ML/results/sep_pret_001.xlsx"
)
end = time.perf_counter()
print(f"{end - start:.4f} seconds = {((end - start)/60):.4f} minutes")