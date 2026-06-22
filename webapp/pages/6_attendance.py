import streamlit as st
import os
import glob
import pandas as pd
import sys
from datetime import datetime

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from auth.auth import require_auth

st.set_page_config(page_title="Attendance History", page_icon="📅")
require_auth()

st.title("Attendance History")

project_root = os.path.dirname(webapp_dir)
data_dir = os.path.join(project_root, "data", "processed")
csv_files = glob.glob(os.path.join(data_dir, "session_*.csv"))

if not csv_files:
    st.info("No attendance records found.")
    st.stop()

username = st.session_state['username']
records = []

for f in csv_files:
    try:
        filename = os.path.basename(f)
        # Parse timestamp from filename: session_YYYY-MM-DD_HH-MM-SS.csv
        date_str = filename.replace("session_", "").replace(".csv", "")
        dt = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
        
        df = pd.read_csv(f)
        if 'person' in df.columns:
            user_df = df[df['person'] == username]
            if not user_df.empty:
                duration_sec = user_df['time'].max() - user_df['time'].min()
                
                real_frames = user_df[user_df['state'] != 'spoof']
                focus_pct = 0
                if len(real_frames) > 0:
                    focused = len(real_frames[real_frames['state'] == 'focused'])
                    focus_pct = (focused / len(real_frames)) * 100
                
                records.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Time": dt.strftime("%H:%M:%S"),
                    "Duration (min)": round(duration_sec / 60, 2),
                    "Focus Score (%)": round(focus_pct, 1)
                })
    except Exception:
        pass

if records:
    df_records = pd.DataFrame(records).sort_values(by=["Date", "Time"], ascending=[False, False])
    st.dataframe(df_records, use_container_width=True)
else:
    st.info("You haven't attended any lectures yet.")
