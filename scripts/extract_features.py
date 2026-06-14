"""
Feature Dataset Extractor Script
==================================
Iterates over labelled image folders, extracts facial features
(EAR, Yaw, Pitch, Roll, Gaze) using MediaPipe, and saves a CSV.

Expected folder layout:
    data/dataset/
    ├── focused/       ← PNG/JPG images of focused students
    └── distracted/    ← PNG/JPG images of distracted students

Usage:
    python scripts/extract_features.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# ── Make src/ importable ─────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features.extractor import FeatureExtractor


class FeatureDatasetExtractor:
    """
    Extracts a feature vector from every image in a labelled dataset
    and saves the results to a CSV file.

    Args:
        dataset_dir:   Root directory with 'focused' and 'distracted' sub-folders.
        output_csv:    Destination path for the resulting CSV.
        landmarker_path: Path to the MediaPipe face_landmarker.task model file.
    """

    LABELS = ["focused", "distracted"]
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, dataset_dir: str, output_csv: str, landmarker_path: str):
        self.dataset_dir = dataset_dir
        self.output_csv = output_csv
        self._extractor = FeatureExtractor()
        self._landmarker = self._build_landmarker(landmarker_path)

    @staticmethod
    def _build_landmarker(model_path: str):
        """Create a MediaPipe FaceLandmarker in IMAGE running mode."""
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
        )
        return FaceLandmarker.create_from_options(options)

    def run(self) -> None:
        """Extract features from all images and save to CSV."""
        rows = []

        for label in self.LABELS:
            folder = os.path.join(self.dataset_dir, label)
            if not os.path.exists(folder):
                print(f"[FeatureDatasetExtractor] Skipping '{label}' — folder not found.")
                continue

            files = [f for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS]

            print(f"[FeatureDatasetExtractor] Processing {len(files)} '{label}' images…")

            for i, filename in enumerate(files):
                image_path = os.path.join(folder, filename)
                features = self._extract_from_image(image_path)
                if features is not None:
                    rows.append(list(features) + [label])
                if i % 100 == 0:
                    print(f"  [{label}] {i}/{len(files)} processed")

        if not rows:
            print("[FeatureDatasetExtractor] No features extracted. Check your dataset directory.")
            return

        cols = FeatureExtractor.FEATURE_NAMES + ["label"]
        df = pd.DataFrame(rows, columns=[c.lower() for c in cols])
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        print(f"\n[FeatureDatasetExtractor] Done! {len(df)} samples → '{self.output_csv}'")

    def _extract_from_image(self, image_path: str) -> np.ndarray | None:
        """
        Load an image, run MediaPipe detection, and return a feature array.

        Returns:
            np.ndarray of shape (5,), or None if no face was detected.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        return self._extractor.extract(result.face_landmarks[0], w, h)


if __name__ == "__main__":
    extractor = FeatureDatasetExtractor(
        dataset_dir=os.path.join(PROJECT_ROOT, "data", "dataset"),
        output_csv=os.path.join(PROJECT_ROOT, "data", "processed", "features.csv"),
        landmarker_path=os.path.join(PROJECT_ROOT, "models", "face_landmarker.task"),
    )
    extractor.run()