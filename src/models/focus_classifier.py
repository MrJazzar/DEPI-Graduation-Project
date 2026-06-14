"""
Focus Classifier Module
=======================
Loads the pre-trained scikit-learn model and StandardScaler,
applies per-session calibration offsets, and predicts focus state.
"""

import numpy as np
import joblib


class FocusClassifier:
    """
    Wraps the pre-trained Random Forest focus classification model.

    After loading, call `set_calibration_offset()` once per session
    to adjust for the user's individual baseline posture.

    Args:
        model_path:  Path to the saved model (.pkl).
        scaler_path: Path to the saved StandardScaler (.pkl).
    """

    FOCUSED_LABEL = "focused"
    DISTRACTED_LABEL = "distracted"

    def __init__(self, model_path: str, scaler_path: str):
        self.clf = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self._calibration_offset = np.zeros(5)
        print("[FocusClassifier] Model and scaler loaded successfully.")

    def set_calibration_offset(self, offset: np.ndarray) -> None:
        """
        Store a per-session calibration offset to correct for baseline differences.

        Args:
            offset: np.ndarray of shape (5,) representing the difference between
                    the user's neutral posture and the training data baseline.
        """
        self._calibration_offset = offset

    def predict(self, raw_features: np.ndarray) -> dict:
        """
        Predict focus state from raw feature values.

        Applies the calibration offset, scales features, then classifies.

        Args:
            raw_features: np.ndarray of shape (5,) → [ear, yaw, pitch, roll, gaze]

        Returns:
            dict with keys:
                - 'label' (str): 'focused' or 'distracted'
                - 'calibrated' (np.ndarray): features after offset correction
                - 'importances' (dict): feature name → contribution percentage
        """
        calibrated = raw_features - self._calibration_offset
        X_scaled = self.scaler.transform([calibrated])
        pred_class = self.clf.predict(X_scaled)[0]
        label = self.FOCUSED_LABEL if pred_class == 1 else self.DISTRACTED_LABEL
        importances = self._get_importances()
        return {
            "label": label,
            "calibrated": calibrated,
            "importances": importances,
        }

    def _get_importances(self) -> dict:
        """Extract feature importances from the Random Forest model (if available)."""
        feature_names = ["EAR", "YAW", "PITCH", "ROLL", "GAZE"]
        if not hasattr(self.clf, "feature_importances_"):
            return {n: 0.0 for n in feature_names}
        imp = self.clf.feature_importances_
        total = imp.sum()
        return {feature_names[i]: round((imp[i] / total) * 100, 1) for i in range(len(feature_names))}
