import os
import pickle
import numpy as np
import cv2
from deepface import DeepFace

# Required fix for Keras 3 / TF 2.16+ compatibility with DeepFace
os.environ["TF_USE_LEGACY_KERAS"] = "1"

def process_and_save_face(image_np: np.ndarray, username: str) -> tuple[bool, str]:
    """
    Extract face embedding using DeepFace and save to individual and global pkl.
    """
    try:
        # Convert RGB (from Streamlit) to BGR for OpenCV/DeepFace
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        results = DeepFace.represent(
            img_path=img_bgr,
            model_name="Facenet",
            enforce_detection=True
        )
        
        if not results:
            return False, "No face detected in the image."
            
        embedding = results[0]["embedding"]
        
        # Paths
        webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(webapp_dir)
        
        # 1. Save individual embedding
        embeddings_dir = os.path.join(webapp_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        individual_path = os.path.join(embeddings_dir, f"{username}.pkl")
        
        with open(individual_path, "wb") as f:
            pickle.dump({username: embedding}, f)
            
        # 2. Update global embedding for the AI system
        global_path = os.path.join(project_root, "data", "processed", "embeddings.pkl")
        os.makedirs(os.path.dirname(global_path), exist_ok=True)
        
        global_embeddings = {}
        if os.path.exists(global_path):
            with open(global_path, "rb") as f:
                global_embeddings = pickle.load(f)
                
        global_embeddings[username] = embedding
        
        with open(global_path, "wb") as f:
            pickle.dump(global_embeddings, f)
            
        return True, individual_path
        
    except Exception as e:
        return False, f"Error processing face: {str(e)}"
