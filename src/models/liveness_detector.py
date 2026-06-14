"""
Liveness Detector Module
========================
Detects whether a face belongs to a real person or a spoofed image
using **blink detection as the sole criterion**.

Logic:
    - After a grace period, if no blink has been detected → SPOOF.
    - A single confirmed blink is enough to mark the person as LIVE.
    - Once proven live, the status stays LIVE (no re-verification needed).
    - If no blink is seen for a long stretch, the status resets to require
      a fresh blink (handles someone swapping a real face for a photo).
"""

import numpy as np
from collections import deque


class LivenessDetector:
    """
    Blink-only liveness detector.

    Why blink-only?
        - Photos and printed faces **never** blink.
        - Head-pose / micro-jitter signals can be faked by simply
          moving the photo in front of the camera.
        - Blink is the simplest, most reliable anti-spoof signal
          that requires no extra ML model.

    Input features (from FeatureExtractor):
        [ear, yaw, pitch, roll, gaze]
        Only index 0 (EAR) is used.
    """

    def __init__(
        self,
        window_size: int = 90,
        relative_drop_pct: float = 0.20,
        absolute_threshold: float = 0.24,
        min_blinks: int = 1,
        grace_samples: int = 18,
        liveness_timeout_samples: int = 150,
    ):
        """
        Args:
            window_size:              Rolling buffer length for EAR values.
            relative_drop_pct:        EAR must drop this % below the rolling
                                      median to count as a blink frame.
            absolute_threshold:       EAR must also be below this value.
            min_blinks:               Blinks required to confirm liveness.
            grace_samples:            Samples to wait before first judgement
                                      (≈6 s @ 3 fps — enough for one blink).
            liveness_timeout_samples: After this many samples without a new
                                      blink, require re-verification.
        """
        # ── Parameters ────────────────────────────────────────────────────
        self._relative_drop_pct = relative_drop_pct
        self._absolute_threshold = absolute_threshold
        self._min_blinks = min_blinks
        self._grace_samples = grace_samples
        self._liveness_timeout = liveness_timeout_samples

        # ── Rolling EAR buffer ────────────────────────────────────────────
        self._ear_buffer = deque(maxlen=window_size)

        # ── Yaw/Pitch kept for backwards-compat logging only ─────────────
        self._yaw_buffer = deque(maxlen=window_size)
        self._pitch_buffer = deque(maxlen=window_size)

        # ── Liveness state ────────────────────────────────────────────────
        self._is_live = True          # optimistic during grace
        self._confidence = 0.0

        # ── Blink counting ────────────────────────────────────────────────
        self._total_blinks = 0
        self._blink_in_progress = False
        self._blink_frame_counter = 0
        self._min_blink_frames = 1    # at least 1 frame of low EAR
        self._max_blink_frames = 10   # reject held-closed eyes (> ~3 s)
        self._samples_since_last_blink = 0

        # ── Total samples counter ────────────────────────────────────────
        self._total_samples = 0

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def update(self, features: np.ndarray) -> None:
        """
        Feed one sample.  features = [ear, yaw, pitch, roll, gaze].
        """
        ear = float(features[0])
        yaw = float(features[1])
        pitch = float(features[2])

        self._ear_buffer.append(ear)
        self._yaw_buffer.append(yaw)
        self._pitch_buffer.append(pitch)
        self._total_samples += 1
        self._samples_since_last_blink += 1

        self._detect_blink()

    @property
    def is_live(self) -> bool:
        return self._is_live

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def status_text(self) -> str:
        if self._total_samples < self._grace_samples:
            return "Checking..."
        if self._is_live:
            return "LIVE"
        return "SPOOF DETECTED"

    def evaluate(self) -> dict:
        """Return a summary dict of the current liveness state."""
        n = self._total_samples
        ear_std = float(np.std(self._ear_buffer)) if len(self._ear_buffer) > 1 else 0.0
        pose_std = self._compute_pose_std()

        # ── Still in grace period ─────────────────────────────────────────
        if n < self._grace_samples:
            return {
                "is_live": True,
                "confidence": 0.0,
                "blink_count": self._total_blinks,
                "ear_std": round(ear_std, 4),
                "pose_std": round(pose_std, 3),
                "status": "Checking...",
            }

        # ── Decision: blink seen? ────────────────────────────────────────
        if self._total_blinks >= self._min_blinks:
            # Check for timeout (long time without a new blink)
            if self._samples_since_last_blink > self._liveness_timeout:
                self._is_live = False
                self._confidence = 0.70
            else:
                self._is_live = True
                self._confidence = 1.0
        else:
            # No blink ever → spoof
            self._is_live = False
            self._confidence = 0.85

        return {
            "is_live": self._is_live,
            "confidence": round(self._confidence, 2),
            "blink_count": self._total_blinks,
            "ear_std": round(ear_std, 4),
            "pose_std": round(pose_std, 3),
            "status": self.status_text,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Blink detection
    # ─────────────────────────────────────────────────────────────────────

    def _detect_blink(self) -> None:
        """
        Detect blinks by looking for a temporary EAR dip below the
        person's own rolling median.

        A blink is confirmed when:
          1. EAR drops below (median × (1 - relative_drop_pct))
             AND below absolute_threshold for ≥ min_blink_frames.
          2. EAR then rises back up (end of the dip).
          3. The dip lasted ≤ max_blink_frames (rejects closed eyes).
        """
        ears = list(self._ear_buffer)
        if len(ears) < 5:
            return

        # Baseline: median of "open-eye" values (exclude near-zero glitches)
        valid_ears = [e for e in ears if e > 0.08]
        if len(valid_ears) < 3:
            return

        median_ear = float(np.median(valid_ears))
        relative_thresh = median_ear * (1.0 - self._relative_drop_pct)

        current_ear = ears[-1]

        is_blink_frame = (
            current_ear < relative_thresh
            and current_ear < self._absolute_threshold
        )

        if is_blink_frame:
            self._blink_frame_counter += 1
            if (
                self._blink_frame_counter >= self._min_blink_frames
                and not self._blink_in_progress
            ):
                self._blink_in_progress = True
        else:
            # EAR rose back up → check if we just finished a valid blink
            if (
                self._blink_in_progress
                and self._min_blink_frames
                    <= self._blink_frame_counter
                    <= self._max_blink_frames
            ):
                self._total_blinks += 1
                self._samples_since_last_blink = 0

            self._blink_in_progress = False
            self._blink_frame_counter = 0

    # ─────────────────────────────────────────────────────────────────────
    # Helpers (kept for logging compatibility)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_pose_std(self) -> float:
        """Combined yaw + pitch standard deviation (for logs only)."""
        if len(self._yaw_buffer) < 3:
            return 0.0
        return float(np.std(self._yaw_buffer)) + float(np.std(self._pitch_buffer))