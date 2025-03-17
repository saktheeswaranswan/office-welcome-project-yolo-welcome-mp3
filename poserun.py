import cv2
import json
import time
import os

# --- Configuration ---
json_path = "/home/sakthees/Documents/yolov5-master/output/pose.json"
total_playback_time = 139.0  # Total playback duration in seconds (2 minutes 19 seconds)
target_width, target_height = 640, 420  # Resolution for display

# --- Load JSON Pose Data ---
try:
    with open(json_path, "r") as f:
        pose_data = json.load(f)
except Exception as e:
    print(f"❌ ERROR loading JSON: {e}")
    exit()

if not pose_data:
    print("❌ JSON file is empty!")
    exit()

num_frames = len(pose_data)
# Compute the interval (in seconds) to display each JSON frame
update_interval = total_playback_time / num_frames
print(f"Number of frames: {num_frames}, update interval: {update_interval:.3f} seconds")

# --- Open Live Webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

# --- Timing control for JSON frame updates ---
last_update = time.time()
json_index = 0

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Unable to capture frame from webcam.")
        break

    # Resize the frame to the target resolution (if needed)
    frame = cv2.resize(frame, (target_width, target_height))

    # Update the JSON frame index based on the update interval
    current_time = time.time()
    if current_time - last_update >= update_interval:
        json_index = (json_index + 1) % num_frames
        last_update = current_time

    # Retrieve current JSON frame data
    current_frame_data = pose_data[json_index]
    frame_id = current_frame_data.get("frame_id", -1)
    video_timestamp = current_frame_data.get("video_timestamp", 0.0)
    
    # Display frame info on the webcam feed
    cv2.putText(frame, f"Frame: {frame_id}  Time: {video_timestamp:.2f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # For each detection, draw only the bounding box and label it with person ID
    for detection in current_frame_data.get("detections", []):
        person_id = detection.get("person_id", -1)
        bounding_boxes = detection.get("bounding_box", [])
        for box in bounding_boxes:
            if len(box) >= 4:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Bounding Box Playback", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()

