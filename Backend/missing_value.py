# traitement des valeurs manquantes
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def missing_values(df,max_missing_pct_predictors=0.20):
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
                if numb_numeric==1 or corr_matrice is None:
                    median=df_cleaned[col].median()
                    df_cleaned[col].fillna(median,inplace=True)
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
                    df_cleaned[col].fillna(median,inplace=True)
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
                    df_cleaned[col].fillna(median,inplace=True)
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
                        df_cleaned[col].fillna(median,inplace=True)
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
                        df_cleaned[col].fillna(median,inplace=True)
                        print(f" Méthode Median(Regression échoué)")
            # traitement colonnes catégorielles
            else:
                mode_value=df_cleaned[col].mode()
                if len(mode_value)>0:
                    df_cleaned[col].fillna(mode_value[0], inplace=True)
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
    scaler_y = StandardScaler()

    # Normalisation des X
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_predict_scaled = scaler_X.transform(X_predict)

    # Normalisation du y
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()


    #ÉTAPE 4: ENTRAÎNEMENT DU MODÈLE
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)

    # Score R² sur l'entraînement
    r2_score = model.score(X_train_scaled, y_train_scaled)
    print(f"R² du modèle: {r2_score:.4f}")

    if r2_score < 0.1:
        print(f"⚠️ R² très faible ({r2_score:.4f}) - Les prédictions peuvent être imprécises")

    #ÉTAPE 5: PRÉDICTION
    predictions_scaled = model.predict(X_predict_scaled)

    #ÉTAPE 6: DÉNORMALISATION
    # Retour à l'échelle originale
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

    print(f"\nPrédictions - Stats:")
    print(f"  Moyenne: {predictions.mean():.2f}")
    print(f"  Min: {predictions.min():.2f}")
    print(f"  Max: {predictions.max():.2f}")

    #ÉTAPE 7: REMPLACEMENT DANS LE DATAFRAME
    df_regression.loc[predict_mask, target_col] = predictions

    print(f"✅ Imputation par régression réussie pour '{target_col}'")

    return df_regression
