# traitement des valeurs abberantes
import numpy as np


def handle_outliers(df,threshold_iqr=1.5):
    # une valeur aberrante c'est une valeur EXTRÊME qui ne suit pas le modèle général.
    df_cleaned = df.copy()
    initial_count = len(df_cleaned)

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
                'n_outliers': n_outliers,
                'percentage': pourcentage,
                'lower_bound': round(lower_bound,2),
                'upper_bound': round(upper_bound,2),
                'action': 'Suppression',
                'outlier_values': outlier_values.tolist()[:10]
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
                'n_outliers': n_outliers,
                'percentage': pourcentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_capped_below': n_below,
                'n_capped_above': n_above,
                'action': 'Capping',
                'outlier_values': outlier_values.tolist()[:10]
            })

    # suppression des lignes
    if len(rows_to_remove) > 0:
        df_cleaned = df_cleaned.drop(index=list(rows_to_remove))
        df_cleaned = df_cleaned.reset_index(drop=True)

    stats['final_count'] = len(df_cleaned)
    stats['rows_removed'] = len(rows_to_remove)

    return df_cleaned, stats