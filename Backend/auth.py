from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from Backend.database import get_db_connection
from mysql.connector import Error


# ═══════════════════════════════════════════
# MODÈLE UTILISATEUR (Flask-Login)
# ═══════════════════════════════════════════
class User(UserMixin):
    def __init__(self, id, nom, prenom, email, role):
        self.id     = id
        self.nom    = nom
        self.prenom = prenom
        self.email  = email
        self.role   = role

    def is_admin(self):
        return self.role == 'admin'

    def get_full_name(self):
        return f"{self.prenom} {self.nom}"


# ═══════════════════════════════════════════
# CHARGEMENT UTILISATEUR (requis par Flask-Login)
# ═══════════════════════════════════════════
def load_user(user_id):
    """Charge un utilisateur par son ID — appelé par Flask-Login"""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        if row:
            return User(row['id'], row['nom'], row['prenom'], row['email'], row['role'])
        return None
    except Error:
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# ═══════════════════════════════════════════
# INSCRIPTION
# ═══════════════════════════════════════════
def register_user(nom, prenom, email, password):
    """
    Inscrit un nouvel utilisateur.
    Retourne (True, user_id) ou (False, message_erreur)
    """
    conn = get_db_connection()
    if not conn:
        return False, "Connexion base de données impossible"

    try:
        cursor = conn.cursor(dictionary=True)

        # Vérifier si l'email existe déjà
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return False, "Cet email est déjà utilisé"

        # Hasher le mot de passe
        password_hash = generate_password_hash(password)

        # Insérer l'utilisateur
        cursor.execute("""
            INSERT INTO users (nom, prenom, email, password_hash, role)
            VALUES (%s, %s, %s, %s, 'user')
        """, (nom, prenom, email, password_hash))
        conn.commit()

        user_id = cursor.lastrowid
        print(f"✅ Utilisateur créé (ID: {user_id})")
        return True, user_id

    except Error as e:
        conn.rollback()
        print(f"❌ Erreur inscription : {e}")
        return False, "Erreur lors de l'inscription"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# ═══════════════════════════════════════════
# CONNEXION
# ═══════════════════════════════════════════
def login_user_auth(email, password):
    """
    Vérifie les identifiants.
    Retourne (User, None) ou (None, message_erreur)
    """
    conn = get_db_connection()
    if not conn:
        return None, "Connexion base de données impossible"

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cursor.fetchone()

        if not row:
            return None, "Email ou mot de passe incorrect"

        if not check_password_hash(row['password_hash'], password):
            return None, "Email ou mot de passe incorrect"

        user = User(row['id'], row['nom'], row['prenom'], row['email'], row['role'])
        print(f"✅ Connexion : {user.get_full_name()} ({user.role})")
        return user, None

    except Error as e:
        print(f"❌ Erreur login : {e}")
        return None, "Erreur lors de la connexion"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# ═══════════════════════════════════════════
# INIT TABLE USERS (appelé dans init_db)
# ═══════════════════════════════════════════
def init_users_table(conn):
    """Crée la table users si elle n'existe pas"""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            nom           VARCHAR(100) NOT NULL,
            prenom        VARCHAR(100) NOT NULL,
            email         VARCHAR(150) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role          ENUM('user', 'admin') DEFAULT 'user',
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Ajouter user_id à files si elle n'existe pas encore
    try:
        cursor.execute("""
            ALTER TABLE files
            ADD COLUMN user_id INT,
            ADD FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        """)
        print("✅ Colonne user_id ajoutée à files")
    except Exception:
        pass  # La colonne existe déjà

    conn.commit()
    print("✅ Table users prête")