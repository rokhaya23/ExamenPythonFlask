from Backend.database import get_db_connection
from mysql.connector import Error
import json


def get_file_details(file_id):
    """Récupère tous les détails d'un fichier traité."""
    conn = get_db_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM files WHERE id = %s", (file_id,))
        file_info = cursor.fetchone()

        if not file_info:
            return None

        # Logs des traitements
        query_logs = """
            SELECT * FROM processing_logs
            WHERE file_id = %s
            ORDER BY timestamp
        """
        cursor.execute(query_logs, (file_id,))
        treatments = cursor.fetchall()

        # Parser les JSON stats
        for treatment in treatments:
            if treatment.get('stats_json'):
                try:
                    treatment['stats'] = json.loads(treatment['stats_json'])
                except json.JSONDecodeError:
                    treatment['stats'] = {}
            else:
                treatment['stats'] = {}

        initial_rows = file_info.get('initial_rows', 0) or 0
        final_rows   = file_info.get('final_rows', 0) or 0
        initial_cols = file_info.get('initial_columns', 0) or 0
        final_cols   = file_info.get('final_columns', 0) or 0

        summary = {
            'total_treatments': len(treatments),
            'rows_removed':     initial_rows - final_rows,
            'columns_changed':  final_cols - initial_cols,
            'success_rate':     (sum(1 for t in treatments if t.get('success')) / len(treatments) * 100) if treatments else 0
        }

        return {
            'file_info':  file_info,
            'treatments': treatments,
            'summary':    summary
        }

    except Error as e:
        print(f"Erreur récupération détails : {e}")
        return None

    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_all_files_summary(user_id=None):
    """
    Récupère la liste des fichiers traités.
    - user_id=None → tous les fichiers (admin)
    - user_id=X    → seulement les fichiers de cet utilisateur
    """
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)

        base_query = """
            SELECT f.id,
                   f.original_filename,
                   f.output_filename,
                   f.upload_date,
                   f.file_extension,
                   f.initial_rows,
                   f.final_rows,
                   f.initial_columns,
                   f.final_columns,
                   f.processing_status,
                   f.user_id,
                   COUNT(pl.id)                                     as treatments_count,
                   SUM(CASE WHEN pl.success = 1 THEN 1 ELSE 0 END) as success_count
            FROM files f
            LEFT JOIN processing_logs pl ON f.id = pl.file_id
        """

        if user_id is None:
            query = base_query + " GROUP BY f.id ORDER BY f.upload_date DESC"
            cursor.execute(query)
        else:
            query = base_query + " WHERE f.user_id = %s GROUP BY f.id ORDER BY f.upload_date DESC"
            cursor.execute(query, (user_id,))

        files = cursor.fetchall()

        for file in files:
            initial_rows      = file.get('initial_rows', 0) or 0
            final_rows        = file.get('final_rows', 0) or 0
            initial_cols      = file.get('initial_columns', 0) or 0
            final_cols        = file.get('final_columns', 0) or 0
            treatments_count  = file.get('treatments_count', 0) or 0
            success_count     = file.get('success_count', 0) or 0

            file['rows_removed']   = initial_rows - final_rows
            file['columns_added']  = final_cols - initial_cols
            file['success_rate']   = (success_count / treatments_count * 100) if treatments_count > 0 else 0

        return files

    except Error as e:
        print(f"Erreur récupération fichiers : {e}")
        return []

    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_statistics(user_id=None):
    """
    Statistiques globales.
    - user_id=None → stats de tous les fichiers (admin)
    - user_id=X    → stats uniquement des fichiers de cet utilisateur
    """
    conn = get_db_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor(dictionary=True)

        if user_id is None:
            query_general = """
                SELECT COUNT(*) as total_files,
                       COALESCE(SUM(initial_rows), 0) as total_rows_initial,
                       COALESCE(SUM(final_rows), 0)   as total_rows_final,
                       CAST(COALESCE(AVG(
                           CASE WHEN initial_rows > 0
                                THEN (final_rows * 100.0 / initial_rows)
                                ELSE 0 END
                       ), 0) AS DECIMAL(10,2)) as avg_retention_rate
                FROM files
                WHERE processing_status = 'success'
            """
            cursor.execute(query_general)
        else:
            query_general = """
                SELECT COUNT(*) as total_files,
                       COALESCE(SUM(initial_rows), 0) as total_rows_initial,
                       COALESCE(SUM(final_rows), 0)   as total_rows_final,
                       CAST(COALESCE(AVG(
                           CASE WHEN initial_rows > 0
                                THEN (final_rows * 100.0 / initial_rows)
                                ELSE 0 END
                       ), 0) AS DECIMAL(10,2)) as avg_retention_rate
                FROM files
                WHERE processing_status = 'success'
                  AND user_id = %s
            """
            cursor.execute(query_general, (user_id,))

        general = cursor.fetchone()

        if general and general.get('avg_retention_rate') is not None:
            general['avg_retention_rate'] = float(general['avg_retention_rate'])
        else:
            if general:
                general['avg_retention_rate'] = 0.0

        # Traitements les plus utilisés (filtrés par user si besoin)
        if user_id is None:
            query_treatments = """
                SELECT pl.treatment_type,
                       COUNT(*) as usage_count,
                       SUM(CASE WHEN pl.success = 1 THEN 1 ELSE 0 END) as success_count
                FROM processing_logs pl
                WHERE pl.enabled = 1
                GROUP BY pl.treatment_type
                ORDER BY usage_count DESC
            """
            cursor.execute(query_treatments)
        else:
            query_treatments = """
                SELECT pl.treatment_type,
                       COUNT(*) as usage_count,
                       SUM(CASE WHEN pl.success = 1 THEN 1 ELSE 0 END) as success_count
                FROM processing_logs pl
                JOIN files f ON pl.file_id = f.id
                WHERE pl.enabled = 1
                  AND f.user_id = %s
                GROUP BY pl.treatment_type
                ORDER BY usage_count DESC
            """
            cursor.execute(query_treatments, (user_id,))

        treatments = cursor.fetchall()

        return {
            'general':    general,
            'treatments': treatments
        }

    except Error as e:
        print(f"Erreur statistiques : {e}")
        return None

    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()