"""
Feature Extractor Module
========================
Combines all facial feature computations (EAR, Head Pose, Gaze)
into a single, reusable class.
"""

import numpy as np
from .ear import LEFT_EYE, RIGHT_EYE, compute_ear
from .head_pose import get_head_pose
from .gaze import gaze_proxy


class FeatureExtractor:
    """
    Extracts a fixed-size feature vector from MediaPipe FaceLandmarks.

    Features (in order):
        [0] EAR   - Eye Aspect Ratio (drowsiness indicator)
        [1] YAW   - Head horizontal rotation
        [2] PITCH - Head vertical rotation
        [3] ROLL  - Head tilt rotation
        [4] GAZE  - Iris position relative to eye width
    """

    FEATURE_NAMES = ["EAR", "YAW", "PITCH", "ROLL", "GAZE"]

    def extract(self, landmarks, img_w: int, img_h: int) -> np.ndarray:
        """
        Compute all features from a list of FaceLandmark objects.

        Args:
            landmarks: List of landmark objects from MediaPipe FaceLandmarker.
            img_w: Frame width in pixels.
            img_h: Frame height in pixels.

        Returns:
            np.ndarray of shape (5,) → [ear, yaw, pitch, roll, gaze]
        """
        ear = self._compute_ear(landmarks, img_w, img_h)
        yaw, pitch, roll = get_head_pose(landmarks, img_w, img_h)
        gaze = gaze_proxy(landmarks)
        return np.array([ear, yaw, pitch, roll, gaze])

    def _compute_ear(self, landmarks, img_w: int, img_h: int) -> float:
        """Compute the average Eye Aspect Ratio for both eyes."""
        left_eye = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_EYE]
        right_eye = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_EYE]
        return (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
