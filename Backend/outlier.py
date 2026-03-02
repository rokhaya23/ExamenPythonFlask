# traitement des valeurs abberantes
import numpy as np
import pandas as pd


def detect_type_mismatch_outliers(df):
    """
    Détecte et corrige automatiquement les valeurs dont le type ne correspond pas
    au type dominant de la colonne.

    Fonctionne pour tous les types : numérique, texte, etc.
    Pas de liste prédéfinie de valeurs.
    """
    import numpy as np
    import pandas as pd

    df_cleaned = df.copy()
    type_mismatch_log = []

    for col in df_cleaned.columns:
        non_null_values = df_cleaned[col].dropna()

        if len(non_null_values) == 0:
            continue

        # Compter les types : numérique vs non-numérique
        numeric_count = 0
        non_numeric_count = 0

        for val in non_null_values:
            # Essayer de convertir en nombre
            try:
                float(val)
                numeric_count += 1
            except (ValueError, TypeError):
                non_numeric_count += 1

        total = len(non_null_values)

        # Déterminer le type dominant
        if numeric_count > non_numeric_count:
            dominant_type = 'numeric'
            dominant_count = numeric_count
        else:
            dominant_type = 'non_numeric'
            dominant_count = non_numeric_count

        dominant_ratio = dominant_count / total

        # Si un type domine clairement (> 70% des valeurs)
        if dominant_ratio > 0.7:
            outliers_detected = []

            # Parcourir toutes les valeurs et remplacer celles qui ne sont pas du type dominant
            for idx, val in df_cleaned[col].items():
                if pd.isna(val):
                    continue

                # Vérifier si la valeur correspond au type dominant
                is_numeric = False
                try:
                    float(val)
                    is_numeric = True
                except (ValueError, TypeError):
                    is_numeric = False

                val_type = 'numeric' if is_numeric else 'non_numeric'

                # Si le type ne correspond pas au type dominant
                if val_type != dominant_type:
                    outliers_detected.append({
                        'index': idx,
                        'value': val,
                        'type': val_type
                    })
                    # Remplacer par NaN
                    df_cleaned.at[idx, col] = np.nan

            # Afficher les résultats
            if len(outliers_detected) > 0:
                print(f"   ⚠️ '{col}' : {len(outliers_detected)} valeurs de type incorrect")
                print(f"      Type dominant : {dominant_type} ({dominant_ratio * 100:.0f}%)")

                # Afficher quelques exemples
                for item in outliers_detected[:3]:
                    print(f"      → '{item['value']}' (type: {item['type']})")

                if len(outliers_detected) > 3:
                    print(f"      → ... et {len(outliers_detected) - 3} autres")

                type_mismatch_log.append({
                    'column': col,
                    'dominant_type': dominant_type,
                    'dominant_ratio': dominant_ratio,
                    'outliers_count': len(outliers_detected)
                })

    return df_cleaned, type_mismatch_log

def handle_outliers(df,threshold_iqr=1.5):
    # une valeur aberrante c'est une valeur EXTRÊME qui ne suit pas le modèle général.
    df_cleaned = df.copy()
    initial_count = len(df_cleaned)

    #1 : Détecter les valeurs de type incorrect
    print("\n🔍 DÉTECTION DES VALEURS DE TYPE INCORRECT")
    df_cleaned, type_mismatch_log = detect_type_mismatch_outliers(df_cleaned)

    #Convertir les colonnes numériques après correction des types
    df_cleaned = cast_columns_after_mismatch_fix(df_cleaned)

    #2 : Détecter les outliers numériques (IQR)
    print("\n🔍 DÉTECTION DES OUTLIERS NUMÉRIQUES (IQR)")

    #extraire les valeurs numeriques
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    numb_numeric = len(numeric_cols)

    if numb_numeric==0:
        print("Aucune colonne numerique")
        stats = {
            'initial_count': initial_count,
            'final_count': initial_count,
            'rows_removed': 0,
            'numeric_cols': [],
            'outliers_details': []
        }
        return df_cleaned, stats

    # ici set nous permet de conserver les indices a supprimer et eviter les doublons
    #si on a une ligne avec des outliers sur 2colonne elle ne sera compter une fois
    rows_to_remove = set()

    stats={
        'initial_count' : initial_count,
        'numeric_cols' : numeric_cols,
        'outliers_details' : []
    }

    for col in numeric_cols:
        col_lower = col.lower()

        if 'id' in col_lower:
            print(f"   🔑 '{col}' : Colonne ID ignorée")
            stats['outliers_details'].append({
                'column': col,
                'n_outliers': 0,
                'action': 'Ignorée (colonne ID)'
            })
            continue

        #calcul des quartiles
        Q1=df_cleaned[col].quantile(0.25)     # 25% des resultats sont en dessous
        Q3=df_cleaned[col].quantile(0.75)     # 25% des resultats sont en dessous
        IQR = Q3 - Q1     #mesure la dispersion des 50% de valeurs centrales

        #les limites acceptable toute valeur en dehors de ces limites est considérée aberrante
        lower_bound = Q1 - (threshold_iqr * IQR)
        upper_bound = Q3 + (threshold_iqr * IQR)

        # ✅ AJUSTEMENT avec minimum positif
        col_data = df_cleaned[col].dropna()

        if len(col_data) > 0:
            positive_ratio = (col_data > 0).sum() / len(col_data)

            # Si 95%+ des valeurs sont positives
            if positive_ratio > 0.95 and lower_bound < 0:
                positive_values = col_data[col_data > 0]

                if len(positive_values) > 0:
                    min_positive = positive_values.min()
                    print(f"   📊 '{col}' : {positive_ratio * 100:.0f}% positifs")
                    print(f"      Limite ajustée : {lower_bound:.2f} → {min_positive:.2f}")
                    lower_bound = min_positive
        #True si outlier, False sinon, ignorer les valeurs NULL
        outliers_mask=(((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound))& df_cleaned[col].notna())

        #nombre de outliers sur la colonne
        n_outliers = outliers_mask.sum()
        pourcentage = (n_outliers / initial_count)*100

        if n_outliers==0:
            stats['outliers_details'].append({
                'column': col,
                'n_outliers': 0,
                'percentage': 0.0,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'action': 'Aucun outlier détecté'
            })
            continue
        #extraire les valeurs aberrantes
        outlier_values = df_cleaned.loc[outliers_mask, col].values

        if pourcentage < 5:
            # recuperer les indices des lignes contenant les outliers
            outlier_indices = df_cleaned[outliers_mask].index.tolist()
            #rajouter ces indice au niveau du set
            rows_to_remove.update(outlier_indices)

            stats['outliers_details'].append({
                'column': col,
                'n_outliers': int(n_outliers),
                'percentage': float(pourcentage),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'action': 'Suppression',
                'outlier_values': [float(x) for x in outlier_values.tolist()[:10]]
            })
        else:
            # Compter combien de outliers qui sont en dessous et au dessus des limites
            n_below = (df_cleaned[col] < lower_bound).sum()
            n_above = (df_cleaned[col] > upper_bound).sum()

            # Remplacer les valeurs en dessous par la limite basse
            if n_below > 0:
                df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound

            # Remplacer les valeurs au dessus par la limite haute
            if n_above > 0:
                df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound

            stats['outliers_details'].append({
                'column': col,
                'n_outliers': int(n_outliers),
                'percentage': float(pourcentage),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'n_capped_below': int(n_below),
                'n_capped_above': int(n_above),
                'action': 'Capping',
                'outlier_values': [float(x) for x in outlier_values.tolist()[:10]]
            })

    # suppression des lignes
    if len(rows_to_remove) > 0:
        df_cleaned = df_cleaned.drop(index=list(rows_to_remove))
        df_cleaned = df_cleaned.reset_index(drop=True)

    stats['final_count'] = len(df_cleaned)
    stats['rows_removed'] = len(rows_to_remove)

    return df_cleaned, stats


def cast_columns_after_mismatch_fix(df):
    """
    Après avoir remplacé les valeurs de mauvais type par NaN,
    convertit les colonnes à dominante numérique en float.
    """
    df_casted = df.copy()

    for col in df_casted.columns:
        if df_casted[col].dtype == object or pd.api.types.is_string_dtype(df_casted[col]):
            non_null = df_casted[col].dropna()
            if len(non_null) == 0:
                continue

            # Essayer de convertir en numérique
            converted = pd.to_numeric(non_null, errors='coerce')
            success_ratio = converted.notna().sum() / len(non_null)

            # Si 70%+ des valeurs sont convertibles → forcer en float
            if success_ratio >= 0.7:
                df_casted[col] = pd.to_numeric(df_casted[col], errors='coerce')
                print(f"   🔄 '{col}' converti en numérique ({success_ratio * 100:.0f}% de valeurs valides)")

    return df_casted