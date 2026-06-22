import streamlit as st
import os
import glob
import pandas as pd
import sys

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from auth.auth import require_auth

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
require_auth()

st.title("Dashboard")
st.write(f"Welcome back, **{st.session_state['full_name']}**!")

# Calculate metrics from processed data
project_root = os.path.dirname(webapp_dir)
data_dir = os.path.join(project_root, "data", "processed")
csv_files = glob.glob(os.path.join(data_dir, "session_*.csv"))

total_lectures = 0
avg_focus = 0.0
last_lecture_date = "N/A"

if csv_files:
    # Get user's specific data
    username = st.session_state['username']
    user_focus_scores = []
    lectures_attended = 0
    
    # Sort files by modification time descending
    csv_files.sort(key=os.path.getmtime, reverse=True)
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if 'person' in df.columns:
                user_df = df[df['person'] == username]
                if not user_df.empty:
                    lectures_attended += 1
                    # focus is when state == "focused"
                    real_frames = user_df[user_df['state'] != 'spoof']
                    if len(real_frames) > 0:
                        focused_frames = len(real_frames[real_frames['state'] == 'focused'])
                        focus_pct = (focused_frames / len(real_frames)) * 100
                        user_focus_scores.append(focus_pct)
        except Exception:
            pass
            
    total_lectures = lectures_attended
    if user_focus_scores:
        avg_focus = sum(user_focus_scores) / len(user_focus_scores)
    if total_lectures > 0:
        import time
        last_mtime = os.path.getmtime(csv_files[0])
        last_lecture_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(last_mtime))

col1, col2, col3 = st.columns(3)
col1.metric("Total Lectures Attended", total_lectures)
col2.metric("Average Focus Score", f"{avg_focus:.1f}%")
col3.metric("Last Lecture Activity", last_lecture_date)

st.markdown("---")
st.subheader("Quick Actions")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("▶️ Join a Lecture", use_container_width=True):
        st.switch_page("pages/5_join_lecture.py")
with c2:
    if st.button("📈 View Reports", use_container_width=True):
        st.switch_page("pages/7_reports.py")
with c3:
    if st.button("⚙️ Profile Settings", use_container_width=True):
        st.switch_page("pages/8_profile.py")
