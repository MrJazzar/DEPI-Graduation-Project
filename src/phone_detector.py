from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_phone(frame):

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])

            # COCO class 67 = cell phone
            if cls == 67:
                return 1

    return 0