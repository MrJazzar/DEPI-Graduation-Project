import os
import sys
import time
import numpy as np
import cv2
import pickle
from streamlit_webrtc import VideoTransformerBase
from deepface import DeepFace

os.environ["TF_USE_LEGACY_KERAS"] = "1"

webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(webapp_dir)

class RegistrationProcessor(VideoTransformerBase):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.stage_idx = 0
        self.stages = [
            {"name": "FRONT", "desc": "Look Straight"},
            {"name": "RIGHT", "desc": "Turn RIGHT"},
            {"name": "LEFT", "desc": "Turn LEFT"}
        ]
        self.stage_start = time.time()
        self.captured_frames = []
        self.done = False
        self.success = False
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.done:
            cv2.putText(img, "DONE! You can stop the video.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return img
            
        stage = self.stages[self.stage_idx]
        elapsed = time.time() - self.stage_start
        
        if elapsed < 3:
            remain = 3 - int(elapsed)
            cv2.putText(img, f"{stage['desc']} in {remain}...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif elapsed < 5:
            cv2.putText(img, f"Capturing {stage['name']}...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Capture roughly every 0.2 seconds
            if len(self.captured_frames) < (self.stage_idx + 1) * 5:
                self.captured_frames.append(img.copy())
        else:
            self.stage_idx += 1
            self.stage_start = time.time()
            if self.stage_idx >= len(self.stages):
                self.done = True
                self.process_embeddings()
                
        return img
        
    def process_embeddings(self):
        embeddings = []
        for f in self.captured_frames:
            try:
                results = DeepFace.represent(img_path=f, model_name="Facenet", enforce_detection=True)
                if results:
                    embeddings.append(results[0]["embedding"])
            except Exception:
                pass
                
        if not embeddings:
            return
            
        avg_embedding = np.mean(embeddings, axis=0)
        
        embeddings_dir = os.path.join(webapp_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        individual_path = os.path.join(embeddings_dir, f"{self.username}.pkl")
        
        with open(individual_path, "wb") as f:
            pickle.dump({self.username: avg_embedding}, f)
            
        global_path = os.path.join(project_root, "data", "processed", "embeddings.pkl")
        os.makedirs(os.path.dirname(global_path), exist_ok=True)
        
        global_embeddings = {}
        if os.path.exists(global_path):
            with open(global_path, "rb") as f:
                global_embeddings = pickle.load(f)
                
        global_embeddings[self.username] = avg_embedding
        
        with open(global_path, "wb") as f:
            pickle.dump(global_embeddings, f)
            
        self.success = True
