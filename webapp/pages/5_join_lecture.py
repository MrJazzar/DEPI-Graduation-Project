import streamlit as st
import sys
import os
import glob
import pandas as pd
from datetime import datetime, timedelta

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from auth.auth import require_auth
from integration.monitoring_runner import start_ai_session
from integration.n8n import trigger_n8n_webhook
from database.db import get_user_by_username

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
            
            # Trigger n8n webhook for email & Google Sheet
            user_data = get_user_by_username(st.session_state.get('username'))
            if user_data:
                # Replace this URL with your actual n8n webhook URL
                n8n_webhook_url = "https://moamen-aljazzar.app.n8n.cloud/webhook-test/5e589227-75ab-4299-a983-5c1ff3765414"
                # Calculate session statistics
                project_root = os.path.dirname(webapp_dir)
                data_dir = os.path.join(project_root, "data", "processed")
                csv_files = glob.glob(os.path.join(data_dir, "session_*.csv"))
                
                session_stats = {}
                if csv_files:
                    csv_files.sort(key=os.path.getmtime, reverse=True)
                    latest_csv = csv_files[0]
                    
                    try:
                        df = pd.read_csv(latest_csv)
                        user_df = df[df['person'] == user_data['username']]
                        if not user_df.empty:
                            real_frames = user_df[user_df['state'] != 'spoof']
                            n_focused = len(real_frames[real_frames['state'] == 'focused'])
                            total_real = len(real_frames)
                            focus_pct = (n_focused / total_real * 100) if total_real > 0 else 0
                            
                            max_time = user_df['time'].max()
                            session_stats = {
                                "focus_percentage": round(focus_pct, 2),
                                "duration_seconds": round(max_time, 2),
                                "duration_minutes": round(max_time / 60, 2),
                                "spoof_frames": len(user_df[user_df['state'] == 'spoof'])
                            }
                    except Exception as e:
                        print(f"Error parsing session stats: {e}")

                leave_time = datetime.now()
                join_time = leave_time
                if session_stats.get("duration_seconds"):
                    join_time = leave_time - timedelta(seconds=session_stats["duration_seconds"])

                payload = {
                    "username": user_data['username'],
                    "full_name": user_data['full_name'],
                    "email": user_data['email'],
                    "event": "session_completed",
                    "join_time": join_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "leave_time": leave_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "focus_percentage": session_stats.get("focus_percentage", 0),
                    "duration_minutes": session_stats.get("duration_minutes", 0),
                    "spoof_attempts": session_stats.get("spoof_frames", 0)
                }
                
                if n8n_webhook_url != "YOUR_N8N_WEBHOOK_URL_HERE":
                    webhook_success, webhook_msg = trigger_n8n_webhook(n8n_webhook_url, payload)
                    if webhook_success:
                        st.success("✅ Attendance logged and email notification sent via n8n!")
                    else:
                        st.warning(f"⚠️ Could not reach n8n webhook: {webhook_msg}")
                else:
                    st.info("ℹ️ n8n webhook is not configured yet. Add your URL to enable notifications!")

            st.balloons()
            st.info("Check the 'Reports' page to see your focus analytics for this session.")
        else:
            st.error(f"Session error: {msg}")
