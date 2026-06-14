# 🎓 Student Focus Monitoring System

A real-time **Multi-Person Student Monitoring System** built with Python, MediaPipe, DeepFace, YOLOv8, and Scikit-Learn.

The system detects **multiple people** simultaneously, identifies **who each person is**, tracks **whether they are focused or distracted**, detects **phone usage per person**, performs **liveness detection** (anti-spoofing via blink analysis), and generates **per-person analytics** at the end of each session.

---

## ✨ Features

| Feature | Description |
|---|---|
| 👥 Multi-Person Tracking | Tracks up to 4 people simultaneously with per-person state management |
| 👤 Face Recognition | Identifies registered students using DeepFace (Facenet) with cropped face matching |
| 🧠 Focus Classification | Predicts Focused / Distracted per person using a Random Forest model (3 samples/sec majority vote) |
| 📐 Calibration | 3-second baseline calibration per session for accurate predictions |
| 📱 Smart Phone Detection | Detects phones via YOLOv8 and matches each phone to the **nearest person** — only that person is marked distracted |
| 🛡️ Liveness Detection | Anti-spoofing: detects fake faces using dual-threshold adaptive blink detection + head pose analysis |
| 📊 Per-Person Analytics | Line chart (all students) + individual pie charts (Focused / Distracted / Spoof per person) |
| 💾 Timestamped CSV | Each session saves a separate CSV (`session_YYYY-MM-DD_HH-MM-SS.csv`) |

---

## 🗂️ Project Architecture

```
DEPI-Graduation-Project/
│
├── src/                          # Core source code (OOP, modular packages)
│   ├── main.py                   # Entry point -- run this file
│   │
│   ├── monitoring/
│   │   └── camera.py             # CameraMonitor -- calibration + multi-person main loop
│   │
│   ├── features/
│   │   ├── extractor.py          # FeatureExtractor -- EAR, HeadPose, Gaze
│   │   ├── ear.py                # Eye Aspect Ratio utility
│   │   ├── head_pose.py          # Head Pose (Yaw/Pitch/Roll) utility
│   │   └── gaze.py               # Gaze proxy utility
│   │
│   ├── models/
│   │   ├── focus_classifier.py   # FocusClassifier  -- wraps model.pkl + calibration
│   │   ├── face_recognizer.py    # FaceRecognizer   -- DeepFace + cosine similarity
│   │   ├── phone_detector.py     # PhoneDetector    -- YOLOv8 with bbox-based person matching
│   │   ├── liveness_detector.py  # LivenessDetector -- blink + pose anti-spoofing
│   │   └── person_state.py       # PersonState      -- per-person tracking state
│   │
│   ├── analytics/
│   │   └── reporter.py           # SessionReporter -- multi-person CSV + matplotlib charts
│   │
│   └── utils/                    # Shared utilities (reserved)
│
├── scripts/                      # Standalone data-preparation scripts
│   ├── generate_embeddings.py    # EmbeddingGenerator -- video -> embeddings.pkl
│   └── extract_features.py       # FeatureDatasetExtractor -- images -> features.csv
│
├── models/                       # Pre-trained model files (not tracked by git)
│   ├── model.pkl                 # Random Forest focus classifier
│   ├── scaler.pkl                # StandardScaler for features
│   └── face_landmarker.task      # MediaPipe FaceLandmarker model
│
├── data/
│   ├── face_videos/              # Face recognition videos (one per person, empty by default)
│   ├── processed/                # Generated outputs (embeddings.pkl, session CSVs)
│   ├── features_clean.csv        # Pre-extracted training features
│   └── dataset/                  # Focus/Distracted image dataset (from Google Drive)
│       ├── focused/
│       └── distracted/
│
├── notebooks/
│   └── experimentation.ipynb     # Jupyter notebook for model training & analysis
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📥 Dataset Downloads

The training data is hosted on Google Drive. Download and place them as shown below:

| Dataset | Google Drive Link | Place In |
|---|---|---|
| **Training Videos** (raw source videos for the focus dataset) | [Download Videos](https://drive.google.com/drive/folders/17B9R82-H3NcWnOpr7uluK4rOxPRB6Zq7) | Reference only (not needed for running) |
| **Training Images** (labelled Focused / Distracted frames) | [Download Images](https://drive.google.com/drive/folders/1uZdtS8ZCH6qJ5m6MZMQYiab8S1RpYelK) | `data/dataset/focused/` and `data/dataset/distracted/` |

> **Note:** These datasets are only needed to **re-train** the ML model. For normal usage, the pre-trained `model.pkl` and `scaler.pkl` are sufficient.
>
> **Face Recognition Videos** (`data/face_videos/`): This folder is **empty by default**. Each user must record a short video of themselves and place it here to enable face recognition. See Step 4 below.


---

## 🚀 Quick Start (Clone & Run)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DEPI-Graduation-Project.git
cd DEPI-Graduation-Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, YOLOv8 (`yolov8n.pt`) will be downloaded automatically.

### 3. Add Required Model Files

Place the following files (provided separately) into the `models/` folder:

| File | Description |
|---|---|
| `model.pkl` | Pre-trained Random Forest classifier |
| `scaler.pkl` | StandardScaler fitted on the training data |
| `face_landmarker.task` | MediaPipe FaceLandmarker model ([download here](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)) |

### 4. Add Face Videos (for Face Recognition)

Place one video per person in `data/face_videos/`. The filename (without extension) becomes the person's name:

```
data/face_videos/
├── moaz.mp4
├── mohamed.mp4
└── moamen.mp4
```

### 5. Generate Face Embeddings

```bash
py scripts/generate_embeddings.py
```

This reads the videos, extracts face embeddings with DeepFace, and saves them to `data/processed/embeddings.pkl`.

### 6. Run the Monitoring System

```bash
py src/main.py
```

- Look at the camera during the **3-second calibration** phase.
- The system will display each person's **name**, **focus state**, **liveness status**, and **cumulative focus %** live.
- Each person gets their own **bounding box** colored by state (green=focused, orange=distracted, red=spoof).
- If someone holds a **phone**, the system identifies **who** is holding it and marks only that person as distracted.
- If someone holds a **photo** in front of the camera, the system detects it as `SPOOF!!` after ~3 seconds (no blinks detected).
- Press **`q`** to end the session -- per-person analytics charts will appear and a timestamped CSV is saved.

---

## 🔬 (Optional) Re-train the Model

If you want to train the model from scratch:

1. Download the [Training Images](https://drive.google.com/drive/folders/1uZdtS8ZCH6qJ5m6MZMQYiab8S1RpYelK) from Google Drive.
2. Place them in `data/dataset/focused/` and `data/dataset/distracted/`.
3. Extract features:
   ```bash
   py scripts/extract_features.py
   ```
4. Open `notebooks/experimentation.ipynb` to train and evaluate a new model.
5. Save the new `model.pkl` and `scaler.pkl` to the `models/` folder.

---

## 🏗️ System Pipeline

```
Camera Frame
    │
    ├── MediaPipe FaceLandmarker ──► 478 landmarks per face (up to 4 faces)
    │       │
    │       ├── FeatureExtractor ──► [EAR, Yaw, Pitch, Roll, Gaze]
    │       │       │
    │       │       ├── FocusClassifier ──► focused / distracted (per person)
    │       │       └── LivenessDetector ──► live / spoof (per person)
    │       │
    │       └── FaceRecognizer ──► person name (via cropped face + DeepFace)
    │
    ├── YOLOv8 PhoneDetector ──► phone bbox ──► match to nearest person
    │
    └── PersonState (per person) ──► vote buffer + liveness + identity
            │
            └── SessionReporter ──► timestamped CSV + charts
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **MediaPipe** | Face landmark detection (478 points per face) |
| **DeepFace / Facenet** | Face embedding & recognition |
| **YOLOv8** | Real-time phone detection with bounding boxes |
| **Scikit-Learn** | Random Forest focus classifier |
| **TensorFlow** | Backend for DeepFace / MediaPipe |
| **OpenCV** | Camera capture and frame rendering |
| **Matplotlib / Pandas** | Per-person analytics & reporting |
| **NumPy / SciPy** | Feature computation & cosine similarity |
