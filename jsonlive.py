import cv2
import json
import numpy as np

# Load JSON pose data
json_path = "/home/sakthees/Documents/yolov5-master/output/pose.json"

try:
    with open(json_path, "r") as f:
        pose_data = json.load(f)
except Exception as e:
    print(f"❌ ERROR loading JSON: {e}")
    exit()

# Define OpenCV colors (BGR format)
COLOR_KEYPOINTS = (255, 0, 0)     # Blue for keypoints
COLOR_CONNECTIONS = (0, 255, 0)   # Green for connections
THICKNESS = 2

# Define body connections (skeleton) as pairs of indices
BODY_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

cap = cv2.VideoCapture(0)
frame_index = 0
total_frames = len(pose_data)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if total_frames > 0:
        current_pose = pose_data[frame_index]
        keypoints = current_pose.get("keypoints", [])
        valid_keypoints = []

        # Process each keypoint by flattening it
        for kp in keypoints:
            kp = np.array(kp).flatten()  # Ensure kp is a flat array
            if kp.size < 2:
                continue
            x, y = int(kp[0]), int(kp[1])
            valid_keypoints.append((x, y))
            cv2.circle(frame, (x, y), 5, COLOR_KEYPOINTS, -1)

        # Draw connections if possible
        for (i, j) in BODY_CONNECTIONS:
            if i < len(valid_keypoints) and j < len(valid_keypoints):
                cv2.line(frame, valid_keypoints[i], valid_keypoints[j], COLOR_CONNECTIONS, THICKNESS)

        frame_index = (frame_index + 1) % total_frames

    cv2.imshow("Live Pose Estimation", frame)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

