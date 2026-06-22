import sqlite3
import os

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "database.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    with get_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                embedding_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

def create_user(full_name, username, email, password_hash):
    try:
        with get_connection() as conn:
            conn.execute('''
                INSERT INTO users (full_name, username, email, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (full_name, username, email, password_hash))
            return True, "User created successfully"
    except sqlite3.IntegrityError as e:
        return False, "Username or Email already exists."
    except Exception as e:
        return False, str(e)

def get_user_by_username(username):
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return dict(row) if row else None

def update_user_embedding(username, embedding_path):
    with get_connection() as conn:
        conn.execute("UPDATE users SET embedding_path = ? WHERE username = ?", (embedding_path, username))

def update_user_profile(username, new_email, new_password_hash=None):
    with get_connection() as conn:
        if new_password_hash:
            conn.execute("UPDATE users SET email = ?, password_hash = ? WHERE username = ?", 
                         (new_email, new_password_hash, username))
        else:
            conn.execute("UPDATE users SET email = ? WHERE username = ?", 
                         (new_email, username))
