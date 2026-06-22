import streamlit as st
import sys
import os

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from auth.auth import hash_password, is_authenticated
from database.db import create_user

st.set_page_config(page_title="Register", page_icon="📝")

if is_authenticated():
    st.info("You are already logged in.")
    st.stop()

st.title("Register")

with st.form("register_form"):
    full_name = st.text_input("Full Name")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    submitted = st.form_submit_button("Register")
    
    if submitted:
        if not all([full_name, username, email, password]):
            st.error("Please fill all fields.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            username = username.strip().lower()
            pwd_hash = hash_password(password)
            success, msg = create_user(full_name, username, email, pwd_hash)
            if success:
                st.success("Registration successful! Please proceed to Login.")
            else:
                st.error(msg)
