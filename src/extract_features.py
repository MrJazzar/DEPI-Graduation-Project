import os
import cv2
import pandas as pd
import mediapipe as mp

from ear import LEFT_EYE, RIGHT_EYE, compute_ear
from head_pose import get_head_pose
from gaze import gaze_proxy

mp_face_mesh = mp.solutions.face_mesh

DATASET_PATH = "data/dataset"

rows = []

def extract_features(image_path, label, face_mesh):

    image = cv2.imread(image_path)

    if image is None:
        return

    h, w, _ = image.shape

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return

    landmarks = results.multi_face_landmarks[0].landmark


    # -------- EAR --------
    left_eye = []
    right_eye = []

    for idx in LEFT_EYE:
        lm = landmarks[idx]
        left_eye.append(
            (lm.x*w, lm.y*h)
        )

    for idx in RIGHT_EYE:
        lm = landmarks[idx]
        right_eye.append(
            (lm.x*w, lm.y*h)
        )

    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)

    ear = (left_ear + right_ear)/2


    # -------- Head Pose --------
    yaw, pitch, roll = get_head_pose(
        landmarks,
        w,
        h
    )


    # -------- Gaze --------
    gaze = gaze_proxy(landmarks)


    rows.append([
        ear,
        yaw,
        pitch,
        roll,
        gaze,
        label
    ])

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
) as face_mesh:

    for label in ["focused", "distracted"]:

        folder = os.path.join(
            DATASET_PATH,
            label
        )

        files = os.listdir(folder)

        print(
            f"Processing {label} images..."
        )

        for i, file in enumerate(files):

            if file.lower().endswith(
                (".jpg",".jpeg",".png")
            ):

                path = os.path.join(
                    folder,
                    file
                )

                extract_features(
                    path,
                    label,
                    face_mesh
                )

                if i % 100 == 0:
                    print(
                        f"{label}: {i} images processed"
                    )


# ---------- Create CSV ----------
df = pd.DataFrame(
    rows,
    columns=[
        "ear",
        "yaw",
        "pitch",
        "roll",
        "gaze",
        "label"
    ]
)

df.to_csv(
    "data/features.csv",
    index=False
)

print("\nfeatures.csv generated successfully!")
print(
    f"Total samples extracted: {len(df)}"
)