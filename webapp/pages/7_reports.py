import streamlit as st
import os
import glob
import pandas as pd
import sys
import matplotlib.pyplot as plt

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

from auth.auth import require_auth

st.set_page_config(page_title="Focus Reports", page_icon="📈", layout="wide")
require_auth()

st.title("📈 Advanced Focus Reports")

project_root = os.path.dirname(webapp_dir)
data_dir = os.path.join(project_root, "data", "processed")
csv_files = glob.glob(os.path.join(data_dir, "session_*.csv"))
csv_files.sort(key=os.path.getmtime, reverse=True)

username = st.session_state['username']

user_sessions = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        if 'person' in df.columns and not df[df['person'] == username].empty:
            user_sessions.append((f, os.path.basename(f)))
    except Exception:
        pass

if not user_sessions:
    st.info("No reports available. Join a lecture first.")
    st.stop()

st.markdown("---")
col_sel, col_empty = st.columns([1, 2])
with col_sel:
    session_options = {name: path for path, name in user_sessions}
    selected_session = st.selectbox("Select a session to view", list(session_options.keys()))

if selected_session:
    file_path = session_options[selected_session]
    df = pd.read_csv(file_path)
    user_df = df[df['person'] == username].copy()
    
    st.subheader(f"Analytics for {selected_session.replace('session_', '').replace('.csv', '')}")
    
    real_frames = user_df[user_df['state'] != 'spoof']
    n_focused = len(real_frames[real_frames['state'] == 'focused'])
    n_distracted = len(real_frames[real_frames['state'] == 'distracted'])
    n_spoof = len(user_df[user_df['state'] == 'spoof'])
    
    total = n_focused + n_distracted
    focus_pct = (n_focused / total * 100) if total > 0 else 0
    
    distracted_pct = (n_distracted / total * 100) if total > 0 else 0
    total_minutes = (total + n_spoof) / 3 / 60  # 3 FPS assumption
    
    focus_status = "Excellent 🌟" if focus_pct > 80 else ("Good 👍" if focus_pct > 50 else "Needs Improvement ⚠️")
    
    # Stylish Metric Cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Focus Score", f"{focus_pct:.1f}%", focus_status)
    c2.metric("Session Duration", f"{total_minutes:.1f} Mins")
    c3.metric("Distraction Rate", f"{distracted_pct:.1f}%")
    
    st.markdown("---")
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.markdown("### 📉 Focus Percentage Over Time")
        labels = user_df['state'].tolist()
        times = user_df['time'].tolist()
        
        rolling = []
        real_count = 0
        focused_count = 0
        for label in labels:
            if label != "spoof":
                real_count += 1
                if label == "focused":
                    focused_count += 1
            pct = (focused_count / real_count * 100) if real_count > 0 else 0
            rolling.append(pct)
            
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, rolling, color="#1f77b4", linewidth=2)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.7)
        ax.fill_between(times, 0, 50, color="red", alpha=0.1)
        ax.fill_between(times, 50, 100, color="green", alpha=0.1)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Focus (%)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
    with col_chart2:
        st.markdown("### 🍩 Session Distribution")
        
        sizes = []
        pie_labels = []
        colors = []
        explode = []
        
        if n_focused > 0:
            sizes.append(focus_pct)
            pie_labels.append("Focused (%)")
            colors.append("#2ca02c")
            explode.append(0.05)
        if n_distracted > 0:
            sizes.append(distracted_pct)
            pie_labels.append("Distracted (%)")
            colors.append("#d62728")
            explode.append(0.05)
        if n_spoof > 0:
            sizes.append(n_spoof)
            pie_labels.append("Spoof Detected")
            colors.append("#ff7f0e")
            explode.append(0.1)
            
        if sizes:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.pie(sizes, explode=explode, labels=pie_labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=90, textprops={'fontsize': 10})
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig2)
        else:
            st.info("No data to display in pie chart.")
