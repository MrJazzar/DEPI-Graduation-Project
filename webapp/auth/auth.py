import hashlib
import os
import streamlit as st
import sys

# Ensure webapp can be imported
webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from database.db import get_user_by_username

def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    else:
        if isinstance(salt, str):
            salt = bytes.fromhex(salt)
    
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + "$" + pwd_hash.hex()

def verify_password(stored_password, provided_password):
    try:
        salt_hex, hash_hex = stored_password.split("$")
        salt = bytes.fromhex(salt_hex)
        expected_hash = hash_hex
        
        pwd_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return pwd_hash.hex() == expected_hash
    except ValueError:
        return False

def login_user(username, password):
    user = get_user_by_username(username)
    if user and verify_password(user['password_hash'], password):
        st.session_state['logged_in'] = True
        st.session_state['username'] = user['username']
        st.session_state['full_name'] = user['full_name']
        st.session_state['user_id'] = user['id']
        st.session_state['has_face'] = bool(user['embedding_path'])
        return True
    return False

def logout_user():
    keys_to_clear = ['logged_in', 'username', 'full_name', 'user_id', 'has_face']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state['logged_in'] = False

def is_authenticated():
    return st.session_state.get('logged_in', False)

def require_auth():
    if not is_authenticated():
        st.warning("Please log in to access this page.")
        st.stop()
