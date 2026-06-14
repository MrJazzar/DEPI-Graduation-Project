"""
Phone Detector Module
=====================
Wraps YOLOv8 to detect mobile phones in a video frame.
Returns phone bounding boxes so the system can determine
which person is holding the phone.

COCO class ID 67 = 'cell phone'.
"""

from ultralytics import YOLO


class PhoneDetector:
    """
    Detects mobile phones in a given frame using YOLOv8.

    Returns bounding boxes of detected phones, enabling the system
    to match each phone to the nearest person.

    Args:
        model_path: Path to the YOLOv8 .pt weights file.
        confidence: Detection confidence threshold (default 0.4).
    """

    PHONE_CLASS_ID = 67  # COCO class for 'cell phone'

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame) -> bool:
        """
        Run YOLO inference and return True if any phone is found.

        Args:
            frame: BGR NumPy array (OpenCV frame).

        Returns:
            True if a cell phone is detected, False otherwise.
        """
        return len(self.detect_boxes(frame)) > 0

    def detect_boxes(self, frame) -> list:
        """
        Run YOLO inference and return bounding boxes of all detected phones.

        Args:
            frame: BGR NumPy array (OpenCV frame).

        Returns:
            List of (x1, y1, x2, y2) tuples for each detected phone.
        """
        phone_boxes = []
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self.PHONE_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    phone_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return phone_boxes

    @staticmethod
    def match_phone_to_person(phone_box: tuple, people_bboxes: dict) -> str:
        """
        Determine which person is closest to a detected phone.

        Compares the center of the phone bounding box to each person's
        face bounding box center. The closest person is the one holding it.

        Args:
            phone_box:     (x1, y1, x2, y2) of the phone.
            people_bboxes: Dict of {person_id: (x1, y1, x2, y2)} face bboxes.

        Returns:
            person_id of the closest person, or None if no people tracked.
        """
        if not people_bboxes:
            return None

        phone_cx = (phone_box[0] + phone_box[2]) / 2
        phone_cy = (phone_box[1] + phone_box[3]) / 2

        best_pid = None
        best_dist = float("inf")

        for pid, face_box in people_bboxes.items():
            face_cx = (face_box[0] + face_box[2]) / 2
            face_cy = (face_box[1] + face_box[3]) / 2
            dist = ((phone_cx - face_cx) ** 2 + (phone_cy - face_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_pid = pid

        return best_pid