import cv2
import time
import os
import pickle
import numpy as np
from deepface import DeepFace

# Fix for TF 2.16+
os.environ["TF_USE_LEGACY_KERAS"] = "1"

def draw_text(frame, text, x, y, color=(0, 255, 0)):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

def run_video_registration(username):
    webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(webapp_dir)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Failed to open camera."
        
    stages = [
        {"name": "FRONT", "desc": "Look Straight Ahead"},
        {"name": "RIGHT", "desc": "Turn Face Slightly RIGHT"},
        {"name": "LEFT", "desc": "Turn Face Slightly LEFT"}
    ]
    
    captured_frames = []
    
    for stage in stages:
        # Countdown 3 seconds
        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret: break
            
            remain = 3 - int(time.time() - start_time)
            frame_copy = frame.copy()
            draw_text(frame_copy, f"{stage['desc']}", 50, 50, (0, 255, 255))
            draw_text(frame_copy, f"Starting in {remain}...", 50, 100, (0, 165, 255))
            
            cv2.imshow("Face Registration", frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False, "Registration cancelled by user."
                
        # Capture 5 frames over 1 second
        start_time = time.time()
        frames_taken = 0
        while frames_taken < 5 and time.time() - start_time < 2:
            ret, frame = cap.read()
            if not ret: break
            
            frame_copy = frame.copy()
            draw_text(frame_copy, f"Capturing {stage['name']}...", 50, 50, (0, 255, 0))
            
            cv2.imshow("Face Registration", frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False, "Registration cancelled by user."
                
            captured_frames.append(frame.copy())
            frames_taken += 1
            time.sleep(0.2)
            
    cap.release()
    cv2.destroyAllWindows()
    
    if len(captured_frames) == 0:
        return False, "No frames captured."
        
    # Process embeddings
    embeddings = []
    for f in captured_frames:
        try:
            results = DeepFace.represent(img_path=f, model_name="Facenet", enforce_detection=True)
            if results:
                embeddings.append(results[0]["embedding"])
        except Exception:
            pass
            
    if not embeddings:
        return False, "Could not detect a face in the captured frames. Please ensure good lighting and try again."
        
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Save embeddings
    embeddings_dir = os.path.join(webapp_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    individual_path = os.path.join(embeddings_dir, f"{username}.pkl")
    
    with open(individual_path, "wb") as f:
        pickle.dump({username: avg_embedding}, f)
        
    global_path = os.path.join(project_root, "data", "processed", "embeddings.pkl")
    os.makedirs(os.path.dirname(global_path), exist_ok=True)
    
    global_embeddings = {}
    if os.path.exists(global_path):
        with open(global_path, "rb") as f:
            global_embeddings = pickle.load(f)
            
    global_embeddings[username] = avg_embedding
    
    with open(global_path, "wb") as f:
        pickle.dump(global_embeddings, f)
        
    return True, individual_path
