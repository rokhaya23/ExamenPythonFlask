import mysql.connector
from mysql.connector import Error
import json

# ═══ CONFIGURATION DB ═══
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'data_cleaning_app',
    'port': 3306,
    'charset': 'utf8mb4'
}

#connection a la base de données
def get_db_connection():
    """Crée une connexion à MySQL"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Erreur connexion DB : {e}")
        return None

#sauvegarde des fichiers
def save_to_db(action,data):
    conn = get_db_connection()
    if not conn:
        return None if action in ['file', 'log'] else False

    try:
        cursor = conn.cursor()

        #enregistrer un file
        if action == 'file':
            query = """
                    INSERT INTO files
                    (original_filename, saved_filename, file_extension, file_size,
                     initial_rows, initial_columns, processing_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) \
                    """
            values = (
                data.get('original_filename'),
                data.get('saved_filename'),
                data.get('file_extension'),
                data.get('file_size'),
                data.get('initial_rows'),
                data.get('initial_columns'),
                data.get('status', 'processing')
            )
            cursor.execute(query, values)
            conn.commit()
            file_id = cursor.lastrowid
            print(f"Fichier enregistré (ID: {file_id})")
            return file_id

        #enregistrer un log
        elif action == 'log':
            query = """
                    INSERT INTO processing_logs
                        (file_id, treatment_type, enabled, success, stats_json)
                    VALUES (%s, %s, %s, %s, %s) \
                    """
            values = (
                data.get('file_id'),
                data.get('treatment'),
                1 if data.get('enabled', True) else 0,
                1 if data.get('success', True) else 0,
                json.dumps(data.get('stats', {}), ensure_ascii=False, default=str)
            )
            cursor.execute(query, values)
            conn.commit()
            log_id = cursor.lastrowid
            print(f"Log enregistré (ID: {log_id})")
            return log_id

        #mise a jour fichier
        elif action == 'update':
            query = """
                    UPDATE files
                    SET output_filename   = %s,
                        final_rows        = %s,
                        final_columns     = %s,
                        processing_status = %s,
                        error_message     = %s
                    WHERE id = %s \
                    """
            values = (
                data.get('output_filename'),
                data.get('final_rows'),
                data.get('final_columns'),
                data.get('status', 'success'),
                data.get('error_message'),
                data.get('file_id')
            )
            cursor.execute(query, values)
            conn.commit()
            print(f"Statut mis à jour (ID: {data.get('file_id')})")
            return True

        else:
            print(f"Action inconnue : {action}")
            return None

    except Error as e:
        print(f"Erreur sauvegarde : {e}")
        conn.rollback()
        return None if action in ['file', 'log'] else False

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

#recuperation historique file
def get_history(limit=10):
    #Récupère l'historique des fichiers traités et recupere la Liste des fichiers avec leurs infos
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)

        query = """
                SELECT f.id, \
                       f.original_filename, \
                       f.upload_date, \
                       f.initial_rows, \
                       f.final_rows, \
                       f.processing_status, \
                       COUNT(pl.id) as treatments_count
                FROM files f
                         LEFT JOIN processing_logs pl ON f.id = pl.file_id
                GROUP BY f.id
                ORDER BY f.upload_date DESC
                    LIMIT %s \
                """

        cursor.execute(query, (limit,))
        return cursor.fetchall()

    except Error as e:
        print(f"Erreur récupération : {e}")
        return []

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

