import streamlit as st
import sys
import os

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from auth.auth import require_auth
from integration.monitoring_runner import start_ai_session

st.set_page_config(page_title="Join Lecture", page_icon="🎓", layout="wide")
require_auth()

st.title("🎓 Join Lecture (Live Monitor)")

if not st.session_state.get('has_face'):
    st.error("You cannot join a lecture without registering your face first.")
    st.info("Please go to the 'Face Registration' page.")
    st.stop()

st.info("When you are ready to join the lecture, click the button below. This will launch the secure monitoring environment.")
st.warning("⚠️ **Note:** To exit the lecture, press the **'q'** key on your keyboard while focused on the camera window.")

if st.button("🚀 Enter Lecture Room", type="primary", use_container_width=True):
    with st.spinner("Initializing monitoring session... The camera window will appear shortly."):
        # We start the session using the runner. The runner blocks until 'q' is pressed.
        success, msg = start_ai_session()
        
        if success:
            st.success("Session concluded successfully!")
            st.balloons()
            st.info("Check the 'Reports' page to see your focus analytics for this session.")
        else:
            st.error(f"Session error: {msg}")
