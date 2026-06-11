def gaze_proxy(landmarks):

    # Left iris center
    iris = landmarks[468]

    # Left eye corners
    left_corner = landmarks[33]
    right_corner = landmarks[133]

    eye_width = right_corner.x - left_corner.x

    if eye_width == 0:
        return 0

    gaze_x = (iris.x - left_corner.x) / eye_width

    return gaze_x