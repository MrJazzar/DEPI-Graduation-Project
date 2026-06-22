import streamlit as st
import sys
import os

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from auth.auth import require_auth
from database.db import update_user_embedding
from integration.registration_runner import run_video_registration

st.set_page_config(page_title="Face Registration", page_icon="📷")
require_auth()

st.title("Face Registration (Video Mode)")

if st.session_state.get('has_face'):
    st.success("You have already registered your face. You can re-register below if you want to update it.")
else:
    st.warning("Please register your face to participate in lectures.")

st.info("""
**How Video Registration Works:**
1. Click **Start Video Registration** below.
2. An external camera window will open.
3. Follow the instructions on the screen: look **FRONT**, **RIGHT**, and **LEFT**.
4. The window will close automatically, and your secure Face ID will be generated!
""")

if st.button("🚀 Start Video Registration", type="primary", use_container_width=True):
    with st.spinner("Recording your face multi-angles and building your secure Face ID... Please wait."):
        success, msg = run_video_registration(st.session_state['username'])
        
        if success:
            update_user_embedding(st.session_state['username'], msg)
            st.session_state['has_face'] = True
            st.success("Face ID registered successfully!")
            st.balloons()
        else:
            st.error(msg)
