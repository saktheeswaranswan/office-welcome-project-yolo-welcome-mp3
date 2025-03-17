import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

# Ensure output directory exists
output_dir = "/home/sakthees/Documents/yolov5-master/output"
os.makedirs(output_dir, exist_ok=True)

# Define the JSON file path
json_path = os.path.join(output_dir, "pose.json")

# Load YOLO Pose Model
model = YOLO("yolo11m-pose.pt")

# Open video source (0 for webcam or video file)
cap = cv2.VideoCapture("poll.mp4")

pose_data = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else np.array([])
        bbox = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None

        if keypoints.size > 0:
            frame_data = {
                "frame_id": frame_count,
                "keypoints": keypoints.tolist(),
                "bounding_box": bbox.tolist() if bbox is not None else None
            }

            pose_data.append(frame_data)

            # Debug: Print to verify data before writing
            print(f"Saving frame {frame_count} data to {json_path}...")

            # Write to JSON file live
            try:
                with open(json_path, "w") as f:
                    json.dump(pose_data, f, indent=4)
                print("✅ JSON updated successfully!")
            except Exception as e:
                print(f"❌ ERROR saving JSON: {e}")

    frame_count += 1
    cv2.imshow("YOLO Pose Estimation - Biomechanics", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Final JSON saved at: {json_path}")

