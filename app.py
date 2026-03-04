import datetime
import json
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from flask import Flask, render_template, send_file, jsonify, request
from werkzeug.utils import secure_filename

# Depuis Backend
from Backend.duplicate import duplicated_rows
from Backend.file_utils import allowed_file, load_file, save_dataframe_to_file
from Backend.missing_value import missing_values
from Backend.outlier import handle_outliers
from Backend.normalization import normalize
from Backend.database import save_to_db, get_history, get_db_connection, init_db
from Backend.dashboard import get_statistics
from Backend.dashboard import get_all_files_summary
from Backend.dashboard import get_file_details

app = Flask(__name__)


# Configuration
app.config['UPLOADED_FOLDER'] = 'uploads'
app.config['OUTPUTS_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024       # taille max 20MB
app.config['ALLOWED_EXTENSIONS'] = {'xlsx','csv','json','xml','sql'}       # extension fichiers acceptés


#---------------------------------------------------------------------------------------------
# ROUTES
#---------------------------------------------------------------------------------------------
@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_file():
    #validation file
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
        return jsonify({'error': 'Format de fichier non supporté'}), 400

    try:
        # ─── PRÉPARATION ───
        filename = secure_filename(file.filename)
        original_ext = filename.rsplit('.', 1)[1].lower()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        #save file originale
        saved_filename = f"{timestamp}_{filename}"
        upload_path = os.path.join(app.config['UPLOADED_FOLDER'], saved_filename)
        file.save(upload_path)
        file_size = os.path.getsize(upload_path)
        print(f"📁 Fichier reçu : {filename} ({file_size} bytes)")

        #upload file
        df, struct_stats = load_file(upload_path)
        initial_shape = df.shape

        #enregistrer dans la BDD
        file_id = save_to_db('file', {
            'original_filename': filename,
            'saved_filename': saved_filename,
            'file_extension': original_ext,
            'file_size': file_size,
            'initial_rows': initial_shape[0],
            'initial_columns': initial_shape[1],
            'status': 'processing'
        })

        options = {
            'missing_values': request.form.get('missing_values') == 'true',
            'outliers': request.form.get('outliers') == 'true',
            'duplicates': request.form.get('duplicates') == 'true',
            'normalize': request.form.get('normalize') == 'true'
        }

        #stats
        processing_stats = {
            'original_filename': filename,
            'timestamp': timestamp,
            'initial_rows': int(initial_shape[0]),
            'initial_columns': int(initial_shape[1]),
            'structure_transformation': struct_stats,
            'treatments_applied': []
        }

        # 1. Doublons (si coché)
        if options['duplicates']:
            df, duplicate_stats = duplicated_rows(df)
            processing_stats['treatments_applied'].append({
                'treatment': 'Doublons',
                'enabled': True,
                'stats': duplicate_stats
            })
            # enregistrer dans la BDD
            save_to_db('log', {
                'file_id': file_id,
                'treatment': 'Doublons',
                'enabled': True,
                'success': True,
                'stats': duplicate_stats
            })

        # 2. Outliers (si coché)
        if options['outliers']:
            df, outlier_stats = handle_outliers(df)
            processing_stats['treatments_applied'].append({
                'treatment': 'Valeurs aberrantes',
                'enabled': True,
                'stats': outlier_stats
            })
            # enregistrer dans la BDD
            save_to_db('log', {
                'file_id': file_id,
                'treatment': 'Valeurs aberrantes',
                'enabled': True,
                'success': True,
                'stats': outlier_stats
            })

        # 3. Valeurs manquantes (si coché)
        if options['missing_values']:
            df, missing_stats = missing_values(df,skip_hidden_detection=False)
            processing_stats['treatments_applied'].append({
                'treatment': 'Valeurs manquantes',
                'enabled': True,
                'stats': missing_stats
            })
            #enregistrer dans BDD
            save_to_db('log', {
                'file_id': file_id,
                'treatment': 'Valeurs manquantes',
                'enabled': True,
                'success': True,
                'stats': missing_stats
            })

        # 4. Normalisation (si coché)
        if options['normalize']:
            df, normalize_stats = normalize(df)
            processing_stats['treatments_applied'].append({
                'treatment': 'Normalisation',
                'enabled': True,
                'stats': normalize_stats
            })
            #enregistrer dans la BDD
            save_to_db('log', {
                'file_id': file_id,
                'treatment': 'Normalisation',
                'enabled': True,
                'success': True,
                'stats': normalize_stats
            })

        #stats finale
        final_shape = df.shape
        processing_stats['final_rows'] = int(final_shape[0])
        processing_stats['final_columns'] = int(final_shape[1])

        #sauvegarde fichier nettoyer
        output_filename = f"cleaned_{timestamp}_{filename}"
        output_path = os.path.join(app.config['OUTPUTS_FOLDER'], output_filename)
        save_dataframe_to_file(df, output_path, original_ext)

        #mise à jour la BDD
        save_to_db('update', {
            'file_id': file_id,
            'output_filename': output_filename,
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'status': 'success',
            'error_message': None
        })

        stats_json = json.loads(json.dumps(processing_stats, default=str))
        return jsonify({
            'success': True,
            'message': 'Fichier traité avec succès',
            'file_id': file_id,
            'filename': output_filename,
            'stats': stats_json
        })

    except Exception as e:
        print(f"ERREUR : {str(e)}\n")
        return jsonify({
            'error': f'Erreur lors du traitement: {str(e)}'
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Télécharge le fichier nettoyé"""
    file_path = os.path.join(app.config['OUTPUTS_FOLDER'], filename)

    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'Fichier non trouvé'}), 404

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard/<int:file_id>')
def file_details(file_id):
    return render_template('file_details.html', file_id=file_id)

# ===== ROUTES API POUR LE DASHBOARD =====
@app.route('/api/dashboard/stats')
def api_dashboard_stats():
    """API pour les statistiques globales"""
    stats = get_statistics()
    if stats:
        return jsonify(stats)
    return jsonify({'error': 'Erreur récupération stats'}), 500

@app.route('/api/dashboard/files')
def api_dashboard_files():
    """API pour la liste des fichiers"""
    files = get_all_files_summary()
    return jsonify(files)

@app.route('/api/file/<int:file_id>')
def api_file_details(file_id):
    """API pour les détails d'un fichier"""
    details = get_file_details(file_id)
    if details:
        return jsonify(details)
    return jsonify({'error': 'Fichier non trouvé'}), 404


@app.route('/api/file/<int:file_id>/download', methods=['GET'])
def download_file_format(file_id):
    """
    Génère le fichier dans le format demandé
    """
    try:
        format_requested = request.args.get('format', 'csv')
        print(f"🔍 Téléchargement demandé: file_id={file_id}, format={format_requested}")

        # Connexion BDD
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM files WHERE id = %s", (file_id,))
        file_info = cursor.fetchone()

        if not file_info:
            print(f"❌ Fichier introuvable: id={file_id}")
            cursor.close()
            conn.close()
            return jsonify({'error': 'Fichier non trouvé'}), 404

        print(f"✅ Fichier trouvé: {file_info['original_filename']}")

        # Chemin du fichier nettoyé
        cleaned_path = os.path.join(app.config['OUTPUTS_FOLDER'], file_info['output_filename'])
        print(f"🔍 Chemin fichier nettoyé: {cleaned_path}")

        if not os.path.exists(cleaned_path):
            print(f"❌ Fichier physique introuvable: {cleaned_path}")
            cursor.close()
            conn.close()
            return jsonify({'error': 'Fichier nettoyé introuvable sur le disque'}), 404

        # Charger le DataFrame
        original_ext = file_info['file_extension'].lower()
        print(f"🔍 Chargement DataFrame (format original: {original_ext})")

        try:
            if original_ext == 'csv':
                df = pd.read_csv(cleaned_path)
            elif original_ext in ['xlsx', 'xls']:
                df = pd.read_excel(cleaned_path)
            elif original_ext == 'json':
                df = pd.read_json(cleaned_path)
            elif original_ext == 'xml':
                df = pd.read_xml(cleaned_path)
            else:
                df = pd.read_csv(cleaned_path)

            print(f"✅ DataFrame chargé: {len(df)} lignes, {len(df.columns)} colonnes")

        except Exception as e:
            print(f"❌ Erreur chargement DataFrame: {str(e)}")
            cursor.close()
            conn.close()
            return jsonify({'error': f'Erreur chargement fichier: {str(e)}'}), 500

        # Générer nouveau fichier
        base_name = file_info['original_filename'].rsplit('.', 1)[0]
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"cleaned_{timestamp}_{base_name}.{format_requested}"
        output_path = os.path.join(app.config['OUTPUTS_FOLDER'], new_filename)

        print(f"🔍 Génération fichier: {output_path}")

        # Sauvegarder dans le format demandé
        try:
            save_dataframe_to_file(df, output_path, format_requested)
            print(f"✅ Fichier généré avec succès: {new_filename}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {str(e)}")
            cursor.close()
            conn.close()
            return jsonify({'error': f'Erreur génération fichier: {str(e)}'}), 500

        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'filename': new_filename
        })

    except Exception as e:
        print(f"❌ ERREUR GÉNÉRALE: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

#creation automatique des tables de la base de donnee sinon creee
with app.app_context():
    init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)