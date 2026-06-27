<div align="center">
  
# 🎓 Student Focus Monitoring System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)](https://ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-orange.svg)](https://developers.google.com/mediapipe)
[![DeepFace](https://img.shields.io/badge/DeepFace-Face_Recognition-green.svg)](https://github.com/serengil/deepface)

**An intelligent, real-time Web Application to track, analyze, and report student focus and behavior during online lectures using state-of-the-art Computer Vision.**

</div>

---

## 📖 Overview

The **Student Focus Monitoring System** is a cutting-edge web platform designed for educational environments. Built entirely on Python and **Streamlit WebRTC**, it processes live webcam feeds directly in the browser. 

The AI pipeline simultaneously detects multiple students, recognizes their identities, evaluates their focus level (Focused vs. Distracted), flags mobile phone usage, and ensures liveness to prevent photo spoofing. All session data is stored and visualized in comprehensive analytical reports.

---

## ✨ Key Features

- 🖥️ **Interactive Web Interface:** Seamless real-time video processing inside your browser via Streamlit and WebRTC.
- 🔐 **Secure User Management:** SQLite-powered registration, authentication, and personalized profiles.
- 👥 **Multi-Person Tracking:** Accurately tracks up to 4 people simultaneously with independent state management.
- 👤 **Face Registration & Recognition:** Easily capture your face embedding via webcam. The system recognizes you using DeepFace (Facenet).
- 🧠 **Smart Focus Classification:** Predicts student focus using a custom-trained Random Forest model based on facial landmarks (EAR, Pitch, Yaw, Roll, Gaze).
- 📱 **Context-Aware Phone Detection:** Uses YOLOv8 to detect phones and strictly matches the phone to the nearest person holding it.
- 🛡️ **Anti-Spoofing (Liveness):** Dual-threshold blink detection and head-pose analysis ensure the person is real and not a static photo.
- 📊 **Advanced Analytics:** Real-time metrics and post-session interactive charts (Focus breakdown, session duration, spoof attempts).

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/MrJazzar/DEPI-Graduation-Project.git
cd DEPI-Graduation-Project
```

### 2. Install Dependencies

Ensure you have Python 3.10+ installed. Run the following command:

```bash
pip install -r requirements.txt
```
*(Note: Ultralytics YOLOv8 weights (`yolov8n.pt`) will download automatically on the first run.)*

### 3. Add Required AI Models

Create a `models/` folder in the root directory (if not exists) and place the following files inside:

| File | Purpose |
|---|---|
| `model.pkl` | Pre-trained Random Forest classifier for Focus Detection. ([Download Here](https://drive.google.com/drive/folders/1h2tVsqJurCPQygQ97oXnntjUb_Q4JbaY?usp=sharing)) |
| `scaler.pkl` | StandardScaler fitted on the training dataset. ([Download Here](https://drive.google.com/drive/folders/1h2tVsqJurCPQygQ97oXnntjUb_Q4JbaY?usp=sharing)) |
| `face_landmarker.task` | MediaPipe FaceLandmarker weights. ([Download Here](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)) |

### 4. Launch the Web App

Start the Streamlit server:

```bash
streamlit run webapp/app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🛠️ Application Workflow

1. **Sign Up / Log In:** Create an account to access the dashboard.
2. **Face Registration:** Head to the `Face Registration` page. Look at the camera to let the system extract and securely save your facial embedding.
3. **Join Lecture:** Navigate to the `Join Lecture` page, activate the WebRTC stream, and the AI will begin monitoring your session live.
4. **View Reports:** Once the session ends, check the `Reports` page for detailed performance charts and timestamped CSV logs.

---

## 🏗️ System Architecture & Pipeline

### AI Pipeline
```text
Camera Frame (WebRTC)
    │
    ├── MediaPipe FaceLandmarker ──► 478 landmarks per face (up to 4 faces)
    │       │
    │       ├── FeatureExtractor ──► [EAR, Yaw, Pitch, Roll, Gaze]
    │       │       │
    │       │       ├── FocusClassifier ──► Focused / Distracted (per person)
    │       │       └── LivenessDetector ──► Live / Spoof (per person)
    │       │
    │       └── FaceRecognizer ──► Identity Matching (DeepFace / Cosine Similarity)
    │
    ├── YOLOv8 PhoneDetector ──► Phone Bounding Box ──► Matches to nearest person
    │
    └── PersonState (State Manager) ──► Aggregates votes, liveness, and identity
            │
            └── SessionReporter ──► Generates timestamped CSV & Data Visualizations
```

<details>
<summary><b>📂 View Directory Structure</b></summary>

```text
DEPI-Graduation-Project/
│
├── webapp/                       # Streamlit Web Application
│   ├── app.py                    # Main Entry Point
│   ├── pages/                    # UI Pages (Auth, Dashboard, Reports, etc.)
│   ├── database/                 # SQLite DB schema and logic
│   ├── auth/                     # Hashing & Session management
│   ├── integration/              # WebRTC & background monitoring runners
│   ├── utils/                    # Helper functions (DeepFace generators)
│   └── embeddings/               # Secure user embeddings (.pkl)
│
├── src/                          # Core AI and processing pipelines
│   ├── models/                   # AI Models (Focus, Face, Phone, Liveness)
│   ├── features/                 # Mathematical Extractors (EAR, Pose, Gaze)
│   ├── monitoring/               # Real-time multi-person processing logic
│   └── analytics/                # Report & Chart generator logic
│
├── models/                       # Pre-trained model weights (.pkl, .task)
├── data/                         # CSV session logs, datasets
├── requirements.txt              # Project dependencies
└── .gitignore                    # Ignored files for safe cloning
```
</details>

---

## 💻 Technology Stack

- **Frontend & Web Framework:** [Streamlit](https://streamlit.io), `streamlit-webrtc`
- **Database:** SQLite3
- **Computer Vision:** [OpenCV](https://opencv.org/)
- **Face Landmark & Pose:** [MediaPipe](https://developers.google.com/mediapipe)
- **Face Recognition:** [DeepFace](https://github.com/serengil/deepface) (Facenet)
- **Object Detection:** [YOLOv8](https://ultralytics.com/) (Ultralytics)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Random Forest)
- **Data Analytics:** Pandas, Matplotlib, NumPy

---

<div align="center">
  <i>Developed as a Graduation Project for DEPI</i>
</div>
