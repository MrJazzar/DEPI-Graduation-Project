import cv2
import mediapipe as mp
from ear import LEFT_EYE, RIGHT_EYE, compute_ear
from head_pose import get_head_pose
from gaze import gaze_proxy
from phone_detector import detect_phone

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

frame_count = 0
phone = 0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark

            # EAR Calculation
            left_eye = []
            right_eye = []

            for idx in LEFT_EYE:
                lm = landmarks[idx]
                left_eye.append((lm.x*w, lm.y*h))

            for idx in RIGHT_EYE:
                lm = landmarks[idx]
                right_eye.append((lm.x*w, lm.y*h))

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)

            ear = (left_ear + right_ear)/2

            # Head Pose
            yaw, pitch, roll = get_head_pose(
                landmarks,
                w,
                h
            )

            # Gaze
            gaze = gaze_proxy(landmarks)

            # Phone Detection
            # run YOLO every 10 frames
            if frame_count % 10 == 0:
                phone = detect_phone(frame)

            print(
                "EAR:", round(ear,3),
                "Yaw:", round(yaw,1),
                "Pitch:", round(pitch,1),
                "Roll:", round(roll,1),
                "Gaze:", round(gaze,2),
                "Phone:", phone
            )

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()