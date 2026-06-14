"""
Camera Monitor Module
=====================
Owns the webcam, runs the calibration phase, and drives the
real-time monitoring loop that ties together all sub-systems.

Supports multiple simultaneous faces with per-person tracking.
"""

import os
import time
import numpy as np
import cv2
import mediapipe as mp

from features.extractor import FeatureExtractor
from models.face_recognizer import FaceRecognizer
from models.focus_classifier import FocusClassifier
from models.phone_detector import PhoneDetector
from models.person_state import PersonState
from analytics.reporter import SessionReporter


class CameraMonitor:
    """
    Orchestrates real-time multi-person student focus monitoring.

    Pipeline per frame:
        1. MediaPipe detects all faces (up to max_faces).
        2. Each face is matched to a PersonState using bounding-box proximity.
        3. Face recognition runs periodically to update each person's identity.
        4. Focus classification + liveness detection run per person.
        5. Majority vote determines final label per person.
        6. Overlay + logs are generated per person.

    Args:
        face_recognizer:       FaceRecognizer instance.
        focus_classifier:      FocusClassifier instance.
        phone_detector:        PhoneDetector instance.
        reporter:              SessionReporter instance.
        landmarker_path:       Path to MediaPipe face_landmarker.task model.
        csv_output_path:       Where to save session_focus.csv.
        calib_seconds:         Duration of calibration phase (default 3s).
        recognition_interval:  Seconds between face ID attempts (default 2.0).
        phone_detection_interval: Frames between phone checks (default 20).
        samples_per_second:    Classification samples per second (default 3).
        max_faces:             Maximum number of simultaneous faces (default 4).
    """

    TRAIN_MEANS = np.array([0.230, -0.846, 5.288, 0.046, 0.481])

    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        focus_classifier: FocusClassifier,
        phone_detector: PhoneDetector,
        reporter: SessionReporter,
        landmarker_path: str,
        csv_output_path: str,
        calib_seconds: int = 3,
        recognition_interval: float = 2.0,
        phone_detection_interval: int = 20,
        samples_per_second: int = 3,
        max_faces: int = 4,
    ):
        self._face_rec = face_recognizer
        self._classifier = focus_classifier
        self._phone_det = phone_detector
        self._reporter = reporter
        self._csv_output_path = csv_output_path
        self._calib_seconds = calib_seconds
        self._recognition_interval = recognition_interval
        self._phone_interval = phone_detection_interval
        self._sample_interval = 1.0 / samples_per_second
        self._samples_per_second = samples_per_second

        self._feature_extractor = FeatureExtractor()
        self._landmarker = self._build_landmarker(landmarker_path, max_faces)

        # Per-person state: {person_id: PersonState}
        self._people: dict = {}
        self._next_person_id = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_landmarker(model_path: str, max_faces: int):
        """Create a MediaPipe FaceLandmarker that supports multiple faces."""
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=max_faces,
        )
        return FaceLandmarker.create_from_options(options)

    def _match_face_to_person(self, bbox: tuple, now: float) -> str:
        """
        Match a detected face bounding box to an existing PersonState.
        Uses center-point distance. If no close match, creates a new person.

        Args:
            bbox: (x1, y1, x2, y2) of the detected face.
            now:  Current timestamp.

        Returns:
            The person_id (string key) matched or newly created.
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        best_id = None
        best_dist = float("inf")

        for pid, state in self._people.items():
            # Skip people not seen recently (>3 seconds stale)
            if (now - state.last_seen) > 3.0:
                continue
            sx = (state.bbox[0] + state.bbox[2]) / 2
            sy = (state.bbox[1] + state.bbox[3]) / 2
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        # If close enough to an existing person (<150px), reuse them
        if best_id is not None and best_dist < 150:
            self._people[best_id].bbox = bbox
            self._people[best_id].last_seen = now
            return best_id

        # Otherwise create a new person
        new_id = f"person_{self._next_person_id}"
        self._next_person_id += 1
        new_state = PersonState(vote_buffer_size=self._samples_per_second)
        new_state.bbox = bbox
        new_state.last_seen = now
        self._people[new_id] = new_state
        return new_id

    def _draw_person_overlay(self, frame, state: PersonState, idx: int, holding_phone: bool) -> None:
        """Draw bounding box and status text for one person."""
        x1, y1, x2, y2 = state.bbox
        label = state.last_voted_label
        name = state.display_name
        focus_pct = self._reporter.average_focus_pct(state.name)

        # Box color based on state
        if label == "spoof":
            box_color = (0, 0, 255)       # Red
        elif label == "focused":
            box_color = (0, 255, 0)       # Green
        else:
            box_color = (0, 140, 255)     # Orange

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Text above the box
        text_y = max(y1 - 10, 20)

        if label == "spoof":
            cv2.putText(frame, f"{name} - SPOOF!!", (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, "FAKE FACE!", (x1, text_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            phone_tag = " [PHONE!]" if holding_phone else ""
            status_text = f"{name} - {label.capitalize()} ({focus_pct:.0f}%){phone_tag}"
            cv2.putText(frame, status_text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Liveness status below
            if state.liveness_status == "LIVE":
                cv2.putText(frame, "Live", (x1, text_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            elif state.liveness_status == "LIKELY LIVE":
                cv2.putText(frame, "Likely Live", (x1, text_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
            elif state.liveness_status == "Checking...":
                cv2.putText(frame, "Checking...", (x1, text_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────────────────────────────────────

    def _calibrate(self, cap) -> None:
        """Collect feature samples for calibration (uses first face detected)."""
        print("=" * 55)
        print("  CALIBRATION: Please look straight at the camera.")
        print(f"  Sit normally for {self._calib_seconds} seconds...")
        print("=" * 55)

        samples = []
        start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            remaining = self._calib_seconds - (time.time() - start)

            if result.face_landmarks:
                features = self._feature_extractor.extract(result.face_landmarks[0], w, h)
                samples.append(features)
                cv2.putText(frame, f"Calibrating...  {remaining:.1f}s", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "No face detected!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Student Monitoring System", frame)
            cv2.waitKey(1)

            if remaining <= 0:
                break

        if samples:
            offset = np.mean(samples, axis=0) - self.TRAIN_MEANS
            self._classifier.set_calibration_offset(offset)
            print(f"[CameraMonitor] Calibration done. Offset applied.")
        else:
            print("[CameraMonitor] Calibration failed -- no face detected.")

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Open the webcam, calibrate, then start the multi-person monitoring loop."""
        cap = cv2.VideoCapture(0)
        self._calibrate(cap)

        print("\n[CameraMonitor] Multi-person monitoring started -- press 'q' to stop.\n")
        start_time = time.time()
        frame_idx = 0
        last_sample_time = 0.0
        last_recognition_time = 0.0
        last_liveness_check = 0.0
        phone_boxes = []
        phone_holders = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            now = time.time()
            t = round(now - start_time, 2)
            h, w, _ = frame.shape

            # ── Phone Detection (every N frames) ─────────────────────────────
            if frame_idx % self._phone_interval == 0:
                phone_boxes = self._phone_det.detect_boxes(frame)

            # ── Classification tick (3 samples/sec) ──────────────────────────
            is_sample_tick = (now - last_sample_time) >= self._sample_interval
            if is_sample_tick:
                last_sample_time = now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self._landmarker.detect(mp_image)

                active_pids = []

                # Register all faces and extract features in ONE pass
                face_data = []
                for face_landmarks in result.face_landmarks:
                    bbox = FaceRecognizer.get_face_bbox(face_landmarks, w, h)
                    pid = self._match_face_to_person(bbox, now)
                    state = self._people[pid]
                    raw = self._feature_extractor.extract(face_landmarks, w, h)
                    state.liveness.update(raw)
                    active_pids.append(pid)
                    face_data.append((pid, state, raw))

                # Match phones to nearest person
                phone_holders = set()
                if phone_boxes:
                    people_bboxes = {pid: self._people[pid].bbox for pid in active_pids}
                    for phone_box in phone_boxes:
                        holder_pid = PhoneDetector.match_phone_to_person(phone_box, people_bboxes)
                        if holder_pid:
                            phone_holders.add(holder_pid)
                            print(f"  [PHONE] Matched to {self._people[holder_pid].display_name}")

                # Classify each person
                for pid, state, raw in face_data:
                    if pid in phone_holders:
                        state.vote_buffer.append("distracted")
                        state.last_importances = {}
                    else:
                        prediction = self._classifier.predict(raw)
                        state.vote_buffer.append(prediction["label"])
                        state.last_importances = prediction["importances"]
                    state.last_voted_label = state.majority_vote()

                # Face Recognition (every ~2 seconds)
                if (now - last_recognition_time) >= self._recognition_interval:
                    last_recognition_time = now
                    for pid in active_pids:
                        state = self._people[pid]
                        x1, y1, x2, y2 = state.bbox
                        face_crop = rgb[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            name = self._face_rec.identify_crop(face_crop)
                            state.update_name(name)

                # Liveness Check (every 1 second)
                if (now - last_liveness_check) >= 1.0:
                    last_liveness_check = now
                    for pid in active_pids:
                        state = self._people[pid]
                        liveness_result = state.liveness.evaluate()
                        state.liveness_status = liveness_result["status"]

                        if liveness_result["confidence"] > 0:
                            print(f"  [{state.display_name}] LIVENESS: {liveness_result['status']} "
                                  f"| Blinks: {liveness_result['blink_count']} "
                                  f"| EAR_std: {liveness_result['ear_std']}")

                # Spoof override + Logging
                for pid in active_pids:
                    state = self._people[pid]
                    if state.liveness_status == "SPOOF DETECTED":
                        state.last_voted_label = "spoof"

                    self._reporter.log(t, state.name, state.last_voted_label, state.last_importances)
                    print(f"[{t:6.2f}s] {state.display_name:<12} | {state.last_voted_label.upper()}")

                # Cleanup stale people (unseen for >10 seconds)
                stale_pids = [pid for pid, s in self._people.items() if (now - s.last_seen) > 10.0]
                for pid in stale_pids:
                    del self._people[pid]

            # ── Draw overlays for all tracked people ─────────────────────────
            for idx, (pid, state) in enumerate(self._people.items()):
                # Only draw recently-seen people
                if (now - state.last_seen) < 2.0:
                    is_holding_phone = pid in phone_holders
                    self._draw_person_overlay(frame, state, idx, is_holding_phone)

            if phone_boxes:
                cv2.putText(frame, f"PHONE DETECTED! ({len(phone_boxes)})", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Draw phone bounding boxes
                for pb in phone_boxes:
                    cv2.rectangle(frame, (pb[0], pb[1]), (pb[2], pb[3]), (0, 0, 255), 2)
                    cv2.putText(frame, "Phone", (pb[0], pb[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Show number of tracked people
            active_count = sum(1 for s in self._people.values() if (now - s.last_seen) < 2.0)
            cv2.putText(frame, f"People: {active_count}", (w - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Student Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        # ── End of Session ────────────────────────────────────────────────────
        self._reporter.save(self._csv_output_path)
        self._reporter.show_charts()
