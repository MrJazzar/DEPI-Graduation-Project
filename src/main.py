"""
Student Monitoring System -- Entry Point
========================================
Run this file to start the real-time multi-person focus monitoring session.

Usage:
    py src/main.py

Press 'q' inside the camera window to end the session and view analytics.
"""

import os
import sys
from datetime import datetime

# ── Fix for Keras 3 / TF 2.16+ compatibility with DeepFace ──────────────────
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ── Make sure src/ sub-packages are importable ───────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.face_recognizer import FaceRecognizer
from models.focus_classifier import FocusClassifier
from models.phone_detector import PhoneDetector
from analytics.reporter import SessionReporter
from monitoring.camera import CameraMonitor


def build_paths() -> dict:
    """Return all project-relative paths as an absolute-path dict."""
    project_root = os.path.dirname(SRC_DIR)
    return {
        "model":       os.path.join(project_root, "models", "model.pkl"),
        "scaler":      os.path.join(project_root, "models", "scaler.pkl"),
        "embeddings":  os.path.join(project_root, "data",   "processed", "embeddings.pkl"),
        "landmarker":  os.path.join(project_root, "models", "face_landmarker.task"),
        "session_dir": os.path.join(project_root, "data",   "processed"),
    }


def main() -> None:
    paths = build_paths()

    face_recognizer   = FaceRecognizer(embeddings_path=paths["embeddings"])
    focus_classifier  = FocusClassifier(model_path=paths["model"], scaler_path=paths["scaler"])
    phone_detector    = PhoneDetector()
    reporter          = SessionReporter()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_csv = os.path.join(paths["session_dir"], f"session_{timestamp}.csv")

    monitor = CameraMonitor(
        face_recognizer=face_recognizer,
        focus_classifier=focus_classifier,
        phone_detector=phone_detector,
        reporter=reporter,
        landmarker_path=paths["landmarker"],
        csv_output_path=session_csv,
    )
    monitor.run()


if __name__ == "__main__":
    main()