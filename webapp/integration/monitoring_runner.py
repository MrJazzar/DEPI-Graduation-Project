import sys
import os

def start_ai_session():
    """
    Launches the existing AI monitoring system.
    This will block the Streamlit app until the user presses 'q' in the OpenCV window.
    """
    webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(webapp_dir)
    src_dir = os.path.join(project_root, "src")
    
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        
    try:
        from src.main import main as run_monitoring
        run_monitoring()
        return True, "Session completed successfully."
    except Exception as e:
        return False, f"Failed to run AI session: {str(e)}"
