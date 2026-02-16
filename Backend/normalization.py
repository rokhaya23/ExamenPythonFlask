# normalisation des elements du fichier
import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize(df,exclude_col=None):
    df_normalized = df.copy()

    # extraire les valeurs numeriques
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    numb_numeric = len(numeric_cols)

    if numb_numeric==0:
        print("Aucune colonne numerique")
        return df_normalized

    #exclure les colonnes id de la normalisation
    #exclude_col c est les colonne a exclure
    if exclude_col is None:
        exclude_cols=[]
    else:
        exclude_cols=exclude_col.copy()

    #detecter les colonne id
    for col in numeric_cols:
        col_lower = col.lower()
        is_id_name = col_lower == 'id' or '_id' in col_lower or col_lower.endswith('id')
        is_unique = df_normalized[col].nunique() == len(df_normalized)

        if (is_id_name or is_unique) and col not in exclude_cols:
            exclude_cols.append(col)

        # NOUVELLE DÉTECTION : Colonnes avec nombres de 7+ chiffres
        non_null_values = df_normalized[col].dropna()

        if len(non_null_values) > 0:
            sample_values = non_null_values.head(min(10, len(non_null_values)))

            # Compter combien de chiffres ont les valeurs
            has_long_numbers = []
            for val in sample_values:
                str_val = str(int(abs(val))) if val == int(val) else str(abs(val)).replace('.', '')
                num_digits = len(str_val)
                has_long_numbers.append(num_digits >= 7)

            # Si plus de 50% des valeurs ont 7+ chiffres, exclure la colonne
            if sum(has_long_numbers) / len(has_long_numbers) >= 0.5:
                exclude_cols.append(col)
                print(f"   📱 '{col}' : Nombres longs détectés → EXCLUE")

        # Colonnes à normaliser
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

    if len(cols_to_normalize) == 0:
        return df_normalized, {
            'action': 'Aucune normalisation',
            'reason': 'Toutes les colonnes numériques sont exclues',
            'excluded_cols': exclude_cols
        }

    #stats avant normalisation
    stats_before = {}
    for col in cols_to_normalize:
        stats_before[col] = {
            'mean': df_normalized[col].mean(),
            'std': df_normalized[col].std(),
            'min': df_normalized[col].min(),
            'max': df_normalized[col].max()
        }

    #standardisation
    scaler = StandardScaler()
    df_normalized[cols_to_normalize] = scaler.fit_transform(
        df_normalized[cols_to_normalize]
    )

    #stats apres normalisation
    stats_after = {}
    for col in cols_to_normalize:
        stats_after[col] = {
            'mean': df_normalized[col].mean(),
            'std': df_normalized[col].std(),
            'min': df_normalized[col].min(),
            'max': df_normalized[col].max()
        }

    stats = {
        'method': 'Standardisation',
        'description': 'StandardScaler (moyenne=0, écart-type=1)',
        'total_columns': len(df_normalized.columns),
        'numeric_columns': len(numeric_cols),
        'normalized_columns': cols_to_normalize,
        'n_normalized': len(cols_to_normalize),
        'excluded_columns': exclude_cols,
        'n_excluded': len(exclude_cols),
        'stats_before': stats_before,
        'stats_after': stats_after
    }

    return df_normalized, stats