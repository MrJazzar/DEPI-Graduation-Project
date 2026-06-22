import streamlit as st
import sys
import os

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from auth.auth import require_auth, hash_password
from database.db import get_user_by_username, update_user_profile

st.set_page_config(page_title="Profile", page_icon="⚙️")
require_auth()

st.title("Profile Settings")

user = get_user_by_username(st.session_state['username'])

with st.form("profile_form"):
    st.subheader("Update Information")
    new_email = st.text_input("Email", value=user['email'])
    
    st.markdown("---")
    st.subheader("Change Password (optional)")
    new_password = st.text_input("New Password", type="password", help="Leave blank to keep current password")
    confirm_password = st.text_input("Confirm New Password", type="password")
    
    submitted = st.form_submit_button("Save Changes")
    
    if submitted:
        if new_password and new_password != confirm_password:
            st.error("Passwords do not match!")
        else:
            pwd_hash = hash_password(new_password) if new_password else None
            try:
                update_user_profile(user['username'], new_email, pwd_hash)
                st.success("Profile updated successfully!")
            except Exception as e:
                st.error(f"Failed to update profile: {e}")

st.markdown("---")
st.subheader("Face Registration")
if st.session_state.get('has_face'):
    st.success("✅ Face is registered.")
else:
    st.warning("❌ Face not registered.")
    
if st.button("Update Face Data"):
    st.switch_page("pages/3_face_registration.py")
