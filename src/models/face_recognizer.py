"""
Face Recognizer Module
======================
Uses DeepFace (Facenet) to extract face embeddings and identify
people by comparing against pre-stored per-person embeddings.

Supports both full-frame identification (single face) and
cropped-face identification (multi-person).
"""

import os

import pickle
import numpy as np
from scipy.spatial.distance import cosine
from deepface import DeepFace


class FaceRecognizer:
    """
    Recognizes faces in real-time by comparing embeddings with stored ones.

    Embeddings are generated once via `scripts/generate_embeddings.py`
    and stored as a dict {name: avg_embedding_vector} in a .pkl file.

    Args:
        embeddings_path: Path to the .pkl file containing stored embeddings.
        threshold: Maximum cosine distance to accept a match (default 0.65).
    """

    def __init__(self, embeddings_path: str, threshold: float = 0.65):
        self.threshold = threshold
        self.embeddings: dict = {}
        self._load_embeddings(embeddings_path)

    def _load_embeddings(self, path: str) -> None:
        """Load stored embeddings from a pickle file."""
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.embeddings = pickle.load(f)
            print(f"[FaceRecognizer] Loaded {len(self.embeddings)} face embeddings.")
        else:
            print(f"[FaceRecognizer] Warning: embeddings file not found at '{path}'.")
            print("[FaceRecognizer] Run 'py scripts/generate_embeddings.py' first.")

    def _match_embedding(self, embedding: list) -> str:
        """
        Compare an embedding vector against all stored embeddings.

        Returns:
            Best matching person's name, or 'Unknown'.
        """
        best_match, min_dist = "Unknown", float("inf")
        for name, stored_embedding in self.embeddings.items():
            dist = cosine(embedding, stored_embedding)
            if dist < min_dist:
                min_dist = dist
                best_match = name
        return best_match if min_dist < self.threshold else "Unknown"

    def identify(self, rgb_frame: np.ndarray) -> str:
        """
        Identify a person from an RGB camera frame (full frame, single face).

        Args:
            rgb_frame: NumPy array in RGB format (H x W x 3).

        Returns:
            The recognized person's name, or 'Unknown' if no match found.
        """
        if not self.embeddings:
            return "Unknown"

        try:
            results = DeepFace.represent(
                img_path=rgb_frame,
                model_name="Facenet",
                enforce_detection=False,
            )
            if not results:
                return "Unknown"
            return self._match_embedding(results[0]["embedding"])
        except Exception:
            return "Unknown"

    def identify_crop(self, rgb_crop: np.ndarray) -> str:
        """
        Identify a person from a cropped face region.

        Used for multi-person tracking where each face is cropped
        from the frame using landmark bounding boxes.

        Args:
            rgb_crop: Cropped RGB face image (H x W x 3).

        Returns:
            The recognized person's name, or 'Unknown'.
        """
        if not self.embeddings or rgb_crop.size == 0:
            return "Unknown"

        try:
            results = DeepFace.represent(
                img_path=rgb_crop,
                model_name="Facenet",
                enforce_detection=False,
            )
            if not results:
                return "Unknown"
            return self._match_embedding(results[0]["embedding"])
        except Exception:
            return "Unknown"

    @staticmethod
    def get_face_bbox(landmarks, img_w: int, img_h: int, padding: float = 0.3) -> tuple:
        """
        Compute a padded bounding box from MediaPipe face landmarks.

        Args:
            landmarks: List of MediaPipe NormalizedLandmark objects.
            img_w:     Image width in pixels.
            img_h:     Image height in pixels.
            padding:   Fractional padding around the face (default 0.3 = 30%).

        Returns:
            Tuple (x1, y1, x2, y2) in pixel coordinates, clamped to image bounds.
        """
        xs = [lm.x * img_w for lm in landmarks]
        ys = [lm.y * img_h for lm in landmarks]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Add padding
        pad_w = (x2 - x1) * padding
        pad_h = (y2 - y1) * padding

        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(img_w, int(x2 + pad_w))
        y2 = min(img_h, int(y2 + pad_h))

        return (x1, y1, x2, y2)
