import os
import sys
import time
import numpy as np
import cv2
import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(webapp_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.features.extractor import FeatureExtractor
from src.models.face_recognizer import FaceRecognizer
from src.models.focus_classifier import FocusClassifier
from src.models.phone_detector import PhoneDetector
from src.models.person_state import PersonState
from src.analytics.reporter import SessionReporter

class MonitorProcessor(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        paths = {
            "model":       os.path.join(project_root, "models", "model.pkl"),
            "scaler":      os.path.join(project_root, "models", "scaler.pkl"),
            "embeddings":  os.path.join(project_root, "data",   "processed", "embeddings.pkl"),
            "landmarker":  os.path.join(project_root, "models", "face_landmarker.task"),
        }
        self.face_rec = FaceRecognizer(embeddings_path=paths["embeddings"])
        self.classifier = FocusClassifier(model_path=paths["model"], scaler_path=paths["scaler"])
        self.phone_det = PhoneDetector()
        
        self.reporter = SessionReporter()
        self.feature_extractor = FeatureExtractor()
        
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=paths["landmarker"]),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=4,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        
        self._people = {}
        self._next_person_id = 0
        
        self.start_time = time.time()
        self.frame_idx = 0
        self.last_sample_time = 0.0
        self.last_recognition_time = 0.0
        self.last_liveness_check = 0.0
        self.phone_boxes = []
        self.phone_holders = set()
        
        self._sample_interval = 1.0 / 3
        self._recognition_interval = 2.0
        self._phone_interval = 20
        self._samples_per_second = 3
        
    def _match_face_to_person(self, bbox, now):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        best_id = None
        best_dist = float("inf")

        for pid, state in self._people.items():
            if (now - state.last_seen) > 3.0:
                continue
            sx = (state.bbox[0] + state.bbox[2]) / 2
            sy = (state.bbox[1] + state.bbox[3]) / 2
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        if best_id is not None and best_dist < 150:
            self._people[best_id].bbox = bbox
            self._people[best_id].last_seen = now
            return best_id

        new_id = f"person_{self._next_person_id}"
        self._next_person_id += 1
        new_state = PersonState(vote_buffer_size=self._samples_per_second)
        new_state.bbox = bbox
        new_state.last_seen = now
        self._people[new_id] = new_state
        return new_id
        
    def _draw_person_overlay(self, frame, state, idx, holding_phone):
        x1, y1, x2, y2 = state.bbox
        label = state.last_voted_label
        name = state.display_name
        focus_pct = self.reporter.average_focus_pct(state.name)

        if label == "spoof":
            box_color = (0, 0, 255)
        elif label == "focused":
            box_color = (0, 255, 0)
        else:
            box_color = (0, 140, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        text_y = max(y1 - 10, 20)

        if label == "spoof":
            cv2.putText(frame, f"{name} - SPOOF!!", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            phone_tag = " [PHONE!]" if holding_phone else ""
            status_text = f"{name} - {label.capitalize()} ({focus_pct:.0f}%){phone_tag}"
            cv2.putText(frame, status_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_idx += 1
        now = time.time()
        t = round(now - self.start_time, 2)
        h, w, _ = img.shape
        
        if self.frame_idx % self._phone_interval == 0:
            self.phone_boxes = self.phone_det.detect_boxes(img)
            
        is_sample_tick = (now - self.last_sample_time) >= self._sample_interval
        if is_sample_tick:
            self.last_sample_time = now
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            
            active_pids = []
            face_data = []
            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    bbox = FaceRecognizer.get_face_bbox(face_landmarks, w, h)
                    pid = self._match_face_to_person(bbox, now)
                    state = self._people[pid]
                    raw = self.feature_extractor.extract(face_landmarks, w, h)
                    state.liveness.update(raw)
                    active_pids.append(pid)
                    face_data.append((pid, state, raw))
                
            self.phone_holders = set()
            if self.phone_boxes:
                people_bboxes = {pid: self._people[pid].bbox for pid in active_pids}
                for pb in self.phone_boxes:
                    holder_pid = PhoneDetector.match_phone_to_person(pb, people_bboxes)
                    if holder_pid:
                        self.phone_holders.add(holder_pid)
                        
            for pid, state, raw in face_data:
                if pid in self.phone_holders:
                    state.vote_buffer.append("distracted")
                    state.last_importances = {}
                else:
                    prediction = self.classifier.predict(raw)
                    state.vote_buffer.append(prediction["label"])
                    state.last_importances = prediction["importances"]
                state.last_voted_label = state.majority_vote()
                
            if (now - self.last_recognition_time) >= self._recognition_interval:
                self.last_recognition_time = now
                for pid in active_pids:
                    state = self._people[pid]
                    x1, y1, x2, y2 = state.bbox
                    face_crop = rgb[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        name = self.face_rec.identify_crop(face_crop)
                        state.update_name(name)
                        
            if (now - self.last_liveness_check) >= 1.0:
                self.last_liveness_check = now
                for pid in active_pids:
                    state = self._people[pid]
                    res = state.liveness.evaluate()
                    state.liveness_status = res["status"]
                    
            for pid in active_pids:
                state = self._people[pid]
                if state.liveness_status == "SPOOF DETECTED":
                    state.last_voted_label = "spoof"
                self.reporter.log(t, state.name, state.last_voted_label, state.last_importances)
                
            stale_pids = [pid for pid, s in self._people.items() if (now - s.last_seen) > 10.0]
            for pid in stale_pids:
                del self._people[pid]
                
        for idx, (pid, state) in enumerate(self._people.items()):
            if (now - state.last_seen) < 2.0:
                is_holding = pid in self.phone_holders
                self._draw_person_overlay(img, state, idx, is_holding)
                
        if self.phone_boxes:
            for pb in self.phone_boxes:
                cv2.rectangle(img, (pb[0], pb[1]), (pb[2], pb[3]), (0, 0, 255), 2)
                
        return img
