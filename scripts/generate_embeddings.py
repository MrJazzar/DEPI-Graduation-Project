"""
Embedding Generator Script
===========================
Reads video files from `data/face_videos/`, extracts face embeddings using
DeepFace (Facenet), and saves a single embeddings.pkl file.

Usage:
    python scripts/generate_embeddings.py

Each video's filename (without extension) is used as the person's name.
Example: `data/face_videos/moaz.mp4`  ->  key "moaz" in the embeddings dict.
"""

import os
# Required fix for Keras 3 / TF 2.16+ compatibility with DeepFace
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import pickle
import numpy as np
from deepface import DeepFace


class EmbeddingGenerator:
    """
    Generates and saves averaged face embeddings from video files.

    Each video file in `data_dir` is treated as one person.
    Every N-th frame is sampled; embeddings are averaged for robustness.

    Args:
        data_dir:    Directory containing input video files.
        output_path: Destination path for the embeddings .pkl file.
        frame_interval: Sample one frame every N frames (default 15).
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}

    def __init__(self, data_dir: str, output_path: str, frame_interval: int = 15):
        self.data_dir = data_dir
        self.output_path = output_path
        self.frame_interval = frame_interval

    def run(self) -> None:
        """Process all videos and save the resulting embeddings dict."""
        if not os.path.exists(self.data_dir):
            print(f"[EmbeddingGenerator] ERROR: Directory not found → '{self.data_dir}'")
            return

        embeddings_dict = {}
        video_files = [
            f for f in os.listdir(self.data_dir)
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not video_files:
            print("[EmbeddingGenerator] No video files found in the data directory.")
            return

        for video_file in video_files:
            person_name = os.path.splitext(video_file)[0].lower()
            video_path = os.path.join(self.data_dir, video_file)
            print(f"\n[EmbeddingGenerator] Processing: {video_file}  ->  person: '{person_name}'")

            embedding = self._process_video(video_path)
            if embedding is not None:
                embeddings_dict[person_name] = embedding
                print(f"[EmbeddingGenerator] OK - Embedding saved for '{person_name}'")
            else:
                print(f"[EmbeddingGenerator] FAIL - No faces found in '{video_file}'. Skipping.")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump(embeddings_dict, f)

        print(f"\n[EmbeddingGenerator] Done! {len(embeddings_dict)} embeddings saved to '{self.output_path}'")

    def _process_video(self, video_path: str) -> np.ndarray | None:
        """
        Extract and average face embeddings from a single video.

        Returns:
            Averaged embedding np.ndarray, or None if no face was detected.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        person_embeddings = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_interval == 0:
                try:
                    results = DeepFace.represent(
                        img_path=frame,
                        model_name="Facenet",
                        enforce_detection=True,
                    )
                    if results:
                        person_embeddings.append(results[0]["embedding"])
                except Exception:
                    pass  # Frame had no detectable face — skip silently

            frame_count += 1

        cap.release()
        print(f"  Sampled {len(person_embeddings)} frames with faces.")
        return np.mean(person_embeddings, axis=0) if person_embeddings else None


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    generator = EmbeddingGenerator(
        data_dir=os.path.join(project_root, "data", "face_videos"),
        output_path=os.path.join(project_root, "data", "processed", "embeddings.pkl"),
    )
    generator.run()
