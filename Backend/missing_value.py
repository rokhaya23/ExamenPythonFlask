# traitement des valeurs manquantes
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def detect_hidden_missing_values(df):

    df_cleaned = df.copy()
    replacements_log = []

    for col in df_cleaned.columns:
        non_null_values = df_cleaned[col].dropna()

        if len(non_null_values) == 0:
            continue

        detected_missing = []

        # Pour chaque valeur dans la colonne
        for idx, val in df_cleaned[col].items():
            if pd.isna(val):
                continue

            is_missing = False
            val_str = str(val).strip()

            # ✅ RÈGLE 1 : Valeurs vides ou quasi-vides (≤ 2 caractères non-alphanumériques)
            if len(val_str) <= 2:
                # Vérifier si ce n'est que des caractères spéciaux
                alphanumeric_count = sum(c.isalnum() for c in val_str)
                if alphanumeric_count == 0:
                    is_missing = True

            # ✅ RÈGLE 2 : Valeurs qui semblent être des marqueurs (patterns communs)
            if not is_missing and len(val_str) <= 10:
                if len(set(val_str)) <= 2 and not any(c.isalnum() for c in val_str):
                    is_missing = True

            # Si détecté comme manquant
            if is_missing:
                detected_missing.append({'index': idx, 'value': val})
                df_cleaned.at[idx, col] = np.nan

        # Afficher les résultats
        if len(detected_missing) > 0:
            print(f"   🔍 '{col}' : {len(detected_missing)} valeurs manquantes cachées détectées")

            # Afficher quelques exemples
            for item in detected_missing[:3]:
                print(f"      → '{item['value']}'")

            if len(detected_missing) > 3:
                print(f"      → ... et {len(detected_missing) - 3} autres")

            replacements_log.append({
                'column': col,
                'detected': len(detected_missing),
                'examples': [item['value'] for item in detected_missing[:5]]
            })

    return df_cleaned, replacements_log

def missing_values(df,max_missing_pct_predictors=0.20, skip_hidden_detection=False):

    #Détecter les valeurs manquantes cachées
    if not skip_hidden_detection:
        print("\n🔍 DÉTECTION DES VALEURS MANQUANTES CACHÉES")
        df, hidden_missing_log = detect_hidden_missing_values(df)

    initial_count=len(df)
    # enregistrement avec au moins une valeur manquante return true or false
    df_missing=df.isnull().any(axis=1)
    missing_count=df_missing.sum()
    pourcentage=round(missing_count/initial_count*100,2)

    # cas 1 : aucune valeur manquante
    if missing_count==0:
        stats={
            'initial_rows': initial_count,
            'missing_count': 0,
            'pourcentage': 0.0,
            'action' : 'Aucune valeurs manquantes'
        }
        return df, stats

    # cas 2 : moins de 5% → suppression
    if pourcentage<5:
        # suppression des lignes qui contiennent au moins une valeur manquante
        df_cleaned = df[~df_missing]
        df_cleaned = df_cleaned.reset_index(drop=True)
        rows_removed=initial_count - len(df_cleaned)
        stats={
            'initial_rows': initial_count,
            'final_rows': len(df_cleaned),
            'rows_removed': rows_removed,
            'pourcentage': pourcentage,
            'action' : 'Suppression des valeurs manquantes'
        }
        return df_cleaned, stats
    # cas 3 : ≥ 5% → IMPUTATION
    else:
        df_cleaned=df.copy()
        imputation_details = []

        # Extrait toutes les colonnes numériques
        numeric_cols=df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        numb_numeric=len(numeric_cols)

        # Trie toutes les colonnes pour sortir ceux avec des valeurs manquantes
        cols_with_missing = [
            col for col in df_cleaned.columns
            if df_cleaned[col].isnull().sum() > 0
        ]
        # on les met par ordre croissant
        cols_sorted_by_missing = sorted(
            cols_with_missing,
            key=lambda x: df_cleaned[x].isnull().sum()
        )
        # calcule du matrice de correlation
        if numb_numeric>1:
            corr_matrice=df_cleaned[numeric_cols].corr()
        else:
            corr_matrice=None

        # traitement par ordre de chaque colonne avec des valeurs manquantes
        for col in cols_sorted_by_missing:
            if df_cleaned[col].isnull().sum()==0:
                continue

            # traitement colonnes numériques
            if col in numeric_cols:

                if df_cleaned[col].notna().sum() == 0:
                    print(f"   ⚠️ '{col}' : Colonne entièrement vide, suppression")
                    df_cleaned = df_cleaned.drop(columns=[col])
                    numeric_cols.remove(col)
                    continue

                if numb_numeric==1 or corr_matrice is None:
                    median=df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median)
                    imputation_details.append({
                        'column': col,
                        'methode' : 'Median',
                        'type' : 'Numerique',
                        'action' : 'Imputation par mediane'
                    })
                    continue

                # recalcule du matrice de correlation car certaines colonnes peuvent etre imputées entre temps
                corr_matrice=df_cleaned[numeric_cols].corr()

                # a chaque correlation elle se correle avec elle meme donc on supprime sa correlation
                # sinon lorsque l'on calculera la corr max elle sera toujours a 1
                col_corr=corr_matrice[col].drop(col, errors='ignore')

                if len(col_corr)==0:
                    median=df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median)
                    imputation_details.append({
                        'column': col,
                        'methode' : 'Median',
                        'type' : 'Numerique',
                        'action' : 'Imputation par mediane'
                    })
                    continue

                max_corr=abs(col_corr).max()

                # Identification des prédicteurs disponibles
                predictor_available=[
                    c for c in numeric_cols
                    if c != col and
                    df_cleaned[c].isnull().sum()/ len(df_cleaned) < max_missing_pct_predictors
                ]

                if max_corr<0.3:
                    median=df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median)
                    imputation_details.append({
                        'column': col,
                        'methode' : 'Median',
                        'type' : 'Numerique',
                        'action' : 'Imputation par mediane'
                    })
                elif 0.3<=max_corr<=0.6:
                    try:
                        col_knn=predictor_available.copy()
                        # on ajoute la colonne cible pour qu'on puisse utiliser KNN
                        if col not in col_knn:
                            col_knn.append(col)

                        if len(col_knn)<2:
                            raise ValueError("Pas assez de predicteurs")

                        #Normalisation avant KNN
                        #Pourquoi normaliser ?
                        #-KNN calcule des distances entre points
                        #s'il y a une difference entre les valeurs il peut fausser l'imputation
                        #-StandardScaler met toutes les variables à la même échelle

                        # Création du scaler
                        scaler=StandardScaler()

                        # Extraction des données à normaliser
                        df_normalized=df_cleaned[col_knn].copy()

                        # Remplir temporairement les NaN avec médiane (sinon impossible de normaliser)
                        df_subset = df_normalized.fillna(df_normalized.median())

                        # KNN sur données normaliser
                        knn_imputer = KNNImputer(n_neighbors=5)
                        df_imputed_normalized = pd.DataFrame(
                            knn_imputer.fit_transform(df_subset),
                            columns=col_knn,
                            index=df_normalized.index
                        )

                        # DÉNORMALISATION
                        df_imputed = pd.DataFrame(
                            scaler.inverse_transform(df_imputed_normalized),
                            columns=col_knn,
                            index=df_normalized.index
                        )

                        # Remplacer la colonne cible
                        df_cleaned[col] = df_imputed[col]
                        imputation_details.append({
                            'column': col,
                            'methode' : 'KNN',
                            'type' : 'Numerique',
                            'action' : 'Imputation par knn'
                        })

                    except Exception as e:
                        median=df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median)
                        print(f" Methone Median(KNN échoué)")
                else:
                    try:
                        df_cleaned=impute_by_regression(df_cleaned,col)
                        imputation_details.append({
                            'column': col,
                            'methode' : 'Regression',
                            'type' : 'Numerique',
                            'action' : 'Imputation par regression'
                        })
                    except Exception as e:
                        median=df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median)
                        print(f" Méthode Median(Regression échoué)")
            # traitement colonnes catégorielles
            else:
                mode_value=df_cleaned[col].mode()
                if len(mode_value)>0:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
                    imputation_details.append({
                        'column': col,
                        'methode' : 'Mode',
                        'type' : 'Categoriel',
                        'action' : 'Imputation par mode'
                    })
        stats = {
            'initial_rows': initial_count,
            'final_rows': len(df_cleaned),
            'missing_count': missing_count,
            'pourcentage': pourcentage,
            'action': 'Imputation intelligente',
            'imputation_details': imputation_details
        }

        # Vérification finale : imputation des valeurs restantes
        for col in df_cleaned.columns:
            remaining_nan = df_cleaned[col].isnull().sum()

            if remaining_nan > 0:
                print(f"⚠️ '{col}' : {remaining_nan} valeurs manquantes → imputation finale")

                if col in numeric_cols:
                    median = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median)
                    imputation_details.append({
                        'column': col,
                        'methode': 'Median (fallback)',
                        'type': 'Numerique',
                        'action': 'Imputation finale par médiane'
                    })
                else:
                    mode_value = df_cleaned[col].mode()
                    if len(mode_value) > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
                        imputation_details.append({
                            'column': col,
                            'methode': 'Mode (fallback)',
                            'type': 'Categoriel',
                            'action': 'Imputation finale par mode'
                        })
        df_cleaned = round_to_original_precision(df_cleaned, df)
        return df_cleaned, stats

def impute_by_regression(df,target_col,max_missing_pct_predictors=0.20):
    #Etape 1 : selectionner les prédicteurs

    df_regression=df.copy()
    # On prend toutes les colonnes numériques SAUF la cible
    numeric_cols = df_regression.select_dtypes(include=[np.number]).columns.tolist()

    # Filtrer les prédicteurs selon le taux de valeurs manquantes
    predictor_cols = [
        col for col in numeric_cols
        if col != target_col and
           df_regression[col].isnull().sum() / len(df_regression) < max_missing_pct_predictors
    ]

    if len(predictor_cols) == 0:
        raise ValueError("Aucun prédicteur valide trouvé")

    # Lignes avec target présent
    has_target = df_regression[target_col].notna()

    # Lignes avec target ET tous prédicteurs présents
    fully_complete = (
            has_target &
            df_regression[predictor_cols].notna().all(axis=1)
    )

    n_with_target = has_target.sum()
    n_complete = fully_complete.sum()

    pct_complete = (n_complete / n_with_target * 100) if n_with_target > 0 else 0

    if pct_complete < 30:  # Seuil: 30% de lignes complètes
        print(f"⚠️ Seulement {pct_complete:.0f}% de lignes complètes")
        print(f"   → Imputation temporaire des prédicteurs par médiane")

        # Imputer UNIQUEMENT les prédicteurs pour les lignes d'entraînement
        for col in predictor_cols:
            # Calculer médiane sur les lignes avec target
            median_val = df_regression.loc[has_target, col].median()

            # Remplir les NaN (seulement dans les lignes avec target)
            mask_to_fill = has_target & df_regression[col].isnull()
            df_regression.loc[mask_to_fill, col] = median_val

    #Étape 2: SÉPARATION DONNÉES D'ENTRAÎNEMENT / À PRÉDIRE
    # Lignes COMPLÈTES pour l'entraînement
    train_mask = (
            df_regression[target_col].notna() &
            df_regression[predictor_cols].notna().all(axis=1)
    )

    # Lignes À PRÉDIRE
    predict_mask = (
            df_regression[target_col].isna() &
            df_regression[predictor_cols].notna().all(axis=1)
    )

    n_train = train_mask.sum()
    n_predict = predict_mask.sum()

    if n_train == 0:
        raise ValueError("Pas de données d'entraînement")

    if n_predict == 0:
        print(f"⚠️ Aucune valeur à prédire (peut-être déjà imputées)")
        return df_regression

    # Extraction des données
    X_train = df_regression.loc[train_mask, predictor_cols]
    y_train = df_regression.loc[train_mask, target_col]
    X_predict = df_regression.loc[predict_mask, predictor_cols]


    #ÉTAPE 3: NORMALISATION
    # Important pour la régression si les échelles sont différentes
    scaler_X = StandardScaler()
    #scaler_y = StandardScaler()

    # Normalisation des X
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_predict_scaled = scaler_X.transform(X_predict)

    # Normalisation du y
    y_train_original = y_train.values
    #y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()


    #ÉTAPE 4: ENTRAÎNEMENT DU MODÈLE
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_original)

    # Score R² sur l'entraînement
    r2_score = model.score(X_train_scaled, y_train_original)
    print(f"R² du modèle: {r2_score:.4f}")

    if r2_score < 0.1:
        print(f"⚠️ R² très faible ({r2_score:.4f}) - Les prédictions peuvent être imprécises")

    #ÉTAPE 5: PRÉDICTION
    predictions = model.predict(X_predict_scaled)

    # ÉTAPE 6: VALIDATION ET CLIPPING
    # Vérifier que les prédictions sont raisonnables
    y_min = y_train.min()
    y_max = y_train.max()

    # Définir une plage sûre : [min - 20%, max + 20%]
    range_margin = (y_max - y_min) * 0.2
    safe_min = max(0, y_min - range_margin)  # Minimum à 0
    safe_max = y_max + range_margin

    # Clipper les valeurs hors limites
    predictions_clipped = np.clip(predictions, safe_min, safe_max)

    n_clipped = (predictions != predictions_clipped).sum()
    if n_clipped > 0:
        print(f"   ⚠️ {n_clipped} prédictions hors limites → clippées")

    print(f"\nPrédictions - Stats:")
    print(f"  Moyenne: {predictions.mean():.2f}")
    print(f"  Min: {predictions.min():.2f}")
    print(f"  Max: {predictions.max():.2f}")

    #ÉTAPE 7: REMPLACEMENT DANS LE DATAFRAME
    df_regression.loc[predict_mask, target_col] = predictions_clipped

    print(f"✅ Imputation par régression réussie pour '{target_col}'")

    return df_regression

def round_to_original_precision(df_imputed, df_original):
    """
    Arrondit les colonnes numériques imputées à la précision
    observée dans les données originales.

    Ex: si la colonne avait 0 décimale → arrondi à l'entier
        si la colonne avait 1 décimale → arrondi à 1 décimale
    """
    df_result = df_imputed.copy()

    for col in df_result.columns:
        # Travailler uniquement sur les colonnes numériques
        if col not in df_original.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_result[col]):
            continue

        # Calculer le nb max de décimales dans les données originales
        max_decimals = 0
        for val in df_original[col].dropna():
            try:
                s = str(float(val))
                if '.' in s:
                    dec = len(s.split('.')[1].rstrip('0'))
                    max_decimals = max(max_decimals, dec)
            except (ValueError, TypeError):
                pass

        # Arrondir
        df_result[col] = df_result[col].round(max_decimals)

    return df_result
