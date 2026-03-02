import json
import os
import numpy as np
import pandas as pd
import xmltodict


#verification extension fichier
def allowed_file(filename,allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# chargement des fichiers
def load_file(filepath):
    """Charge un fichier selon son extension"""
    ext = filepath.rsplit('.', 1)[1].lower()

    print(f"📂 Chargement : {filepath}")

    if ext == 'csv':
        df = pd.read_csv(filepath)
    elif ext == 'xlsx':
        df = pd.read_excel(filepath)
    elif ext == 'json':
        df = pd.read_json(filepath)
    elif ext == 'xml':
        try:
            # Essayer pandas d'abord (rapide pour XML simples)
            df = pd.read_xml(filepath)

            # Vérifier si des colonnes sont entièrement NaN
            nan_cols = [col for col in df.columns if df[col].isna().all()]

            if len(nan_cols) > 0:
                # Structure imbriquée détectée, utiliser xmltodict
                print(f"   ⚠️ Structure XML imbriquée détectée ({len(nan_cols)} colonnes vides)")
                df = load_nested_xml(filepath)
        except Exception as e:
            # Si pandas échoue, utiliser directement xmltodict
            print(f"   ⚠️ Utilisation du parser XML avancé")
            df = load_nested_xml(filepath)
    else:
        raise ValueError(f"Extension non supportée: {ext}")

    print(f"✅ Chargé : {len(df)} lignes × {len(df.columns)} colonnes")

    # Aplatir les données semi-structurées
    df_flat, struct_stats = data_structured(df)

    return df_flat, struct_stats


def load_nested_xml(filepath):
    """
    Charge un fichier XML avec structures imbriquées
    Utilise xmltodict pour parser automatiquement
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Convertir XML en dictionnaire
    data_dict = xmltodict.parse(xml_content)

    # Extraire les enregistrements
    root_key = list(data_dict.keys())[0]
    root_data = data_dict[root_key]

    if isinstance(root_data, dict):
        record_key = list(root_data.keys())[0]
        records = root_data[record_key]
    else:
        records = root_data

    if not isinstance(records, list):
        records = [records]

    # Normaliser en DataFrame (aplatit automatiquement)
    df = pd.json_normalize(records, sep='_')

    # Convertir les structures complexes en JSON strings
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(1)
            if len(sample) > 0 and isinstance(sample.iloc[0], (dict, list)):
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )

    return df


def data_structured(df):
    df_flat=df.copy()
    initial_count=len(df_flat.columns)
    stats = {
        'initial_columns': initial_count,
        'transformations': []
    }
    #detection des données semi structurer
    dict_cols=[]
    list_cols=[]
    mixed_cols=[]

    for col in df_flat.columns:
        # Vérifier le type des valeurs
        non_null_values = df_flat[col].dropna()

        if len(non_null_values) == 0:
            continue

        # Échantillon de valeurs pour détecter le type
        sample = non_null_values.iloc[0]

        #dictionnaires
        if isinstance(sample, dict):
            dict_cols.append(col)
            print(f"   • '{col}' : DICTIONNAIRE détecté")

        #listes
        elif isinstance(sample, list):
            list_cols.append(col)
            print(f"   • '{col}' : LISTE détectée")

        #types mixtes
        elif non_null_values.apply(lambda x: isinstance(x, (dict, list))).any():
            mixed_cols.append(col)
            print(f"   • '{col}' : TYPES MIXTES détectés")

    if not (dict_cols or list_cols or mixed_cols):
        print("Aucune colonne problématique détectée")
        stats['transformations'].append({
            'type': 'detection',
            'message': 'Données déjà structurées'
        })
        return df_flat, stats

    #applatir les dics
    if dict_cols:
        for col in dict_cols:
            # EXEMPLE de données dans cette colonne
            sample_value = df_flat[col].dropna().iloc[0]
            print(f"      Exemple de valeur : {sample_value}")
            #
            try:
                # Normaliser les dictionnaires en DataFrame
                nested_df = pd.json_normalize(df_flat[col])
                # Renommer les colonnes pour garder le contexte
                nested_df.columns = [f"{col}_{subcol}" for subcol in nested_df.columns]
                # Afficher les nouvelles colonnes créées
                print(f"      → Créé {len(nested_df.columns)} colonnes : {list(nested_df.columns)}")
                # Supprimer l'ancienne colonne (le dictionnaire)
                df_flat = df_flat.drop(columns=[col])
                # Ajouter les nouvelles colonnes
                nested_df.index = df_flat.index
                df_flat = pd.concat([df_flat, nested_df], axis=1)

                stats['transformations'].append({
                    'type': 'flatten_dict',
                    'column': col,
                    'new_columns': list(nested_df.columns)
                })

            except Exception as e:
                print(f"Erreur lors de l'aplatissement : {e}")
                # En cas d'erreur, convertir en string
                df_flat[col] = df_flat[col].astype(str)
                stats['transformations'].append({
                    'type': 'dict_to_string',
                    'column': col,
                    'reason': str(e)
                })

    #traitement des listes
    if list_cols:
        for col in list_cols:
            sample_value = df_flat[col].dropna().iloc[0] if len(df_flat[col].dropna()) > 0 else None

            if sample_value is not None:
                print(f"      Exemple de valeur : {sample_value}")

            try:
                # Parser les strings JSON
                if sample_value and isinstance(sample_value, str):
                    df_flat[col] = df_flat[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith('[') else x
                    )

                # ✅ NOUVEAU : Aplatir les listes d'objets
                first_list = df_flat[col].dropna().iloc[0] if len(df_flat[col].dropna()) > 0 else None

                if isinstance(first_list, list) and len(first_list) > 0 and isinstance(first_list[0], dict):
                    # C'est une liste de dictionnaires → aplatir différemment
                    print(f"      → Liste de dictionnaires détectée, aplatissement...")

                    # Option 1 : Créer une colonne par élément de liste
                    max_items = df_flat[col].apply(lambda x: len(x) if isinstance(x, list) else 0).max()

                    for i in range(max_items):
                        # Créer une colonne pour chaque item
                        item_df = df_flat[col].apply(
                            lambda x: x[i] if isinstance(x, list) and len(x) > i else {}
                        )

                        # Normaliser chaque item
                        if len(item_df.dropna()) > 0:
                            item_normalized = pd.json_normalize(item_df)
                            item_normalized.columns = [f"{col}_{i + 1}_{subcol}" for subcol in item_normalized.columns]
                            item_normalized.index = df_flat.index

                            df_flat = pd.concat([df_flat, item_normalized], axis=1)

                    # Supprimer la colonne originale
                    df_flat = df_flat.drop(columns=[col])

                    stats['transformations'].append({
                        'type': 'flatten_list_of_dicts',
                        'column': col,
                        'items_count': max_items
                    })
                else:
                    # Liste simple → convertir en JSON string
                    df_flat[col] = df_flat[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, list) and len(x) > 0 else np.nan
                    )
                    print(f"      → Converti en JSON string")

            except Exception as e:
                print(f"      ⚠️ Erreur : {e}")
                df_flat[col] = df_flat[col].astype(str)
    #traitement de mixed_cols
    if mixed_cols:
        for col in mixed_cols:
            # Convertir en string par sécurité
            df_flat[col] = df_flat[col].astype(str)
            stats['transformations'].append({
                'type': 'mixed_to_string',
                'column': col
            })

    #nettoyer les types de données avec protection des colonnes sensibles
    for col in df_flat.columns:
        try:
            # Vérifier si la colonne contient des données à protéger
            sample_values = df_flat[col].dropna().head(10)

            if len(sample_values) == 0:
                continue

            should_protect = False

            # Protection 1 : Colonnes avec caractères non-numériques
            for val in sample_values:
                val_str = str(val)
                # Contient @ (email), lettres majoritaires, ou espaces multiples
                if '@' in val_str or val_str.replace(' ', '').isalpha() or val_str.count(' ') > 1:
                    should_protect = True
                    break

            # Protection 2 : Nombres longs (IDs, téléphones)
            if not should_protect:
                non_null_values = df_flat[col].dropna()
                if len(non_null_values) > 0:
                    sample_check = non_null_values.head(min(10, len(non_null_values)))
                    has_long_numbers = []

                    for val in sample_check:
                        try:
                            str_val = str(int(abs(float(val)))) if pd.notna(val) else ''
                            num_digits = len(str_val)
                            has_long_numbers.append(num_digits >= 7)
                        except:
                            pass

                    # Si 50%+ ont 7+ chiffres, c'est probablement un ID ou téléphone
                    if len(has_long_numbers) > 0 and sum(has_long_numbers) / len(has_long_numbers) >= 0.5:
                        should_protect = True

            # Ne convertir que si pas protégé
            if not should_protect:
                df_flat[col] = pd.to_numeric(df_flat[col], errors='ignore')

        except:
            pass

    #stats
    stats['final_columns'] = len(df_flat.columns)
    stats['columns_added'] = stats['final_columns'] - stats['initial_columns']

    print("TRANSFORMATION TERMINÉE")
    print(f"   Colonnes initiales : {stats['initial_columns']}")
    print(f"   Colonnes finales   : {stats['final_columns']}")
    print(f"   Colonnes ajoutées  : {stats['columns_added']}")

    return df_flat, stats

# sauvegarde fichier apres nettoyage
def save_dataframe_to_file(df, filepath, original_ext, export_formats=None):

    # Si aucun format spécifié, utiliser le format original
    if export_formats is None:
        export_formats = [original_ext]

    saved_files = []
    base_path = filepath.rsplit('.', 1)[0]  # Enlever l'extension

    for fmt in export_formats:
        output_path = f"{base_path}.{fmt}"

        try:
            if fmt == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
                saved_files.append(output_path)

            elif fmt in ['xlsx', 'xls']:
                df.to_excel(output_path, index=False, engine='openpyxl')
                saved_files.append(output_path)

            elif fmt == 'json':
                df.to_json(output_path, orient='records', indent=2, force_ascii=False)
                saved_files.append(output_path)

            elif fmt == 'xml':
                df.to_xml(output_path, index=False)
                saved_files.append(output_path)

            elif fmt == 'sql':
                # Générer un script SQL INSERT
                table_name = os.path.basename(base_path).replace('cleaned_', '')
                sql_content = generate_sql_insert(df, table_name)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(sql_content)
                saved_files.append(output_path)

            elif fmt == 'parquet':
                df.to_parquet(output_path, index=False)
                saved_files.append(output_path)

            print(f"   ✅ Sauvegardé : {output_path}")

        except Exception as e:
            print(f"   ❌ Erreur lors de la sauvegarde {fmt}: {e}")

    return saved_files


def generate_sql_insert(df, table_name='cleaned_data'):
    """
    Génère un script SQL INSERT à partir d'un DataFrame
    """
    import re

    # Nettoyer le nom de table
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)

    # Générer CREATE TABLE
    sql_lines = [f"-- Script SQL généré automatiquement"]
    sql_lines.append(f"-- Table : {table_name}\n")

    sql_lines.append(f"CREATE TABLE IF NOT EXISTS {table_name} (")

    # Colonnes
    column_definitions = []
    for col in df.columns:
        # Nettoyer le nom de colonne
        clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)

        # Déterminer le type SQL
        if df[col].dtype == 'int64':
            sql_type = 'INTEGER'
        elif df[col].dtype == 'float64':
            sql_type = 'FLOAT'
        elif df[col].dtype == 'bool':
            sql_type = 'BOOLEAN'
        elif df[col].dtype == 'datetime64[ns]':
            sql_type = 'DATETIME'
        else:
            sql_type = 'TEXT'

        column_definitions.append(f"    {clean_col} {sql_type}")

    sql_lines.append(",\n".join(column_definitions))
    sql_lines.append(");\n")

    # Générer INSERT statements
    sql_lines.append(f"-- Insertion des données\n")

    for idx, row in df.iterrows():
        values = []
        for col in df.columns:
            val = row[col]

            if pd.isna(val):
                values.append('NULL')
            elif isinstance(val, (int, float, bool)):
                values.append(str(val))
            else:
                # Échapper les apostrophes
                val_str = str(val).replace("'", "''")
                values.append(f"'{val_str}'")

        clean_cols = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
        sql_lines.append(
            f"INSERT INTO {table_name} ({', '.join(clean_cols)}) VALUES ({', '.join(values)});"
        )

    return "\n".join(sql_lines)