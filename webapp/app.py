import streamlit as st
import sys
import os

# Ensure webapp can be imported
webapp_dir = os.path.dirname(os.path.abspath(__file__))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from database.db import init_db
from auth.auth import is_authenticated, logout_user

st.set_page_config(
    page_title="AI Student Monitoring System",
    page_icon="🎓",
    layout="wide"
)

# Initialize database
init_db()

st.title("🎓 Student Monitoring Portal")

if is_authenticated():
    st.sidebar.success(f"Logged in as: {st.session_state['full_name']}")
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()
    st.success("Welcome to the Student Monitoring Portal. Please select an option from the sidebar.")
else:
    st.sidebar.warning("Not logged in")
    st.info("Welcome! Please Register or Login from the sidebar to continue.")
