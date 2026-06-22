import streamlit as st
import sys
import os

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from auth.auth import login_user, is_authenticated

st.set_page_config(page_title="Login", page_icon="🔑")

if is_authenticated():
    st.info("You are already logged in.")
    st.stop()

st.title("Login")

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    submitted = st.form_submit_button("Login")
    
    if submitted:
        username = username.strip().lower()
        if login_user(username, password):
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
