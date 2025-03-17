import os
import cv2
import json
import numpy as np
import time
from ultralytics import YOLO

# --- Setup directories and file paths ---
output_dir = "/home/sakthees/Documents/yolov5-master/output"
os.makedirs(output_dir, exist_ok=True)
json_path = os.path.join(output_dir, "pose.json")

# --- Load YOLO Pose Model ---
model = YOLO("yolo11m-pose.pt")

# --- Open video source ---
# Use "poll.mp4" for a video file; change to 0 for webcam.
cap = cv2.VideoCapture("poll.mp4")

# --- Set output resolution ---
output_width, output_height = 640, 420

# --- Initialize data storage ---
pose_data = []  # List to hold frame-by-frame pose detections.
global_person_id = 0  # Unique ID for each detected person across frames.
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for consistent output
    frame = cv2.resize(frame, (output_width, output_height))
    
    # Get the video timestamp in seconds (from video metadata)
    video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Run YOLO Pose inference on the current frame
    results = model(frame)
    annotated_frame = results[0].plot()

    # Prepare a dictionary for this frame's detections
    frame_dict = {
        "frame_id": frame_count,
        "video_timestamp": video_timestamp,
        "detections": []  # List to store each detected person in this frame.
    }

    # Process each detected result in the current frame
    for result in results:
        # Extract keypoints and bounding box if available
        keypoints = (result.keypoints.xy.cpu().numpy() 
                     if result.keypoints is not None else np.array([]))
        bbox = (result.boxes.xyxy.cpu().numpy() 
                if result.boxes is not None else None)

        # Process only if keypoints were detected
        if keypoints.size > 0:
            # Create a list of dictionaries for keypoints with joint numbering.
            keypoints_list = []
            for joint_idx, kp in enumerate(keypoints):
                # Flatten in case the keypoint is nested
                kp_flat = np.array(kp).flatten()
                if kp_flat.shape[0] < 2:
                    continue
                keypoints_list.append({
                    "joint_id": joint_idx,
                    "x": int(kp_flat[0]),
                    "y": int(kp_flat[1])
                })

            # Build the detection dictionary for this person
            detection = {
                "person_id": global_person_id,
                "keypoints": keypoints_list,
                "bounding_box": bbox.tolist() if bbox is not None else None
            }
            frame_dict["detections"].append(detection)
            global_person_id += 1  # Increment unique person id for each detection

    # Save the frame's data if any detections were made
    if frame_dict["detections"]:
        pose_data.append(frame_dict)

    # --- Update JSON file live ---
    try:
        with open(json_path, "w") as f:
            json.dump(pose_data, f, indent=4)
    except Exception as e:
        print("❌ ERROR saving JSON:", e)

    frame_count += 1
    cv2.imshow("YOLO Pose Estimation - Biomechanics", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()

print("✅ Final JSON saved at:", json_path)

