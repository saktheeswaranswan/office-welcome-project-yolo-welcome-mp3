import torch
import cv2
import numpy as np
import pandas as pd
import pygame
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# Initialize Pygame for Sound Alerts
pygame.mixer.init()
ALERT_SOUND = "alert.mp3"  # Ensure this file exists

# Load YOLO Pose Model (Replace with actual pose weight)
model = YOLO("yolov5_pose.pt")  # Change this to your actual pose model weight

# COCO Keypoint Labels (Standard Key Points for Pose Estimation)
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# CSV File Setup
CSV_DIR = "detections"
Path(CSV_DIR).mkdir(exist_ok=True)  # Create folder if it doesn't exist
CSV_FILE = f"{CSV_DIR}/pose_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Define CSV Columns (Keypoints + Bounding Box)
columns = ["Timestamp", "Class", "X1", "Y1", "X2", "Y2", "Confidence", "Image_Path"]
for keypoint in KEYPOINT_NAMES:
    columns.append(f"{keypoint}_X")
    columns.append(f"{keypoint}_Y")
    columns.append(f"{keypoint}_Confidence")

# Create CSV with headers
pd.DataFrame(columns=columns).to_csv(CSV_FILE, index=False)

# Open Video File
cap = cv2.VideoCapture('video.mp4')  # Change this to your video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO Pose inference
    results = model(frame)

    detections = []
    person_detected = False  # Track if at least one person is detected
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Extract confidence scores
        keypoints = result.keypoints.xy.cpu().numpy()  # Extract key points
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        # Iterate over all detections
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            confidence = float(scores[i])
            class_name = "Person"  # Pose estimation is for humans

            # Extract Keypoints
            keypoint_data = []
            for j in range(len(KEYPOINT_NAMES)):
                keypoint_x, keypoint_y = keypoints[i][j]
                keypoint_conf = confidence  # Use overall confidence
                keypoint_data.extend([keypoint_x, keypoint_y, keypoint_conf])

            # Save cropped image
            crop_filename = f"{CSV_DIR}/cropped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            person_crop = frame[y1:y2, x1:x2]
            cv2.imwrite(crop_filename, person_crop)

            # Play Alert Sound if a person is detected
            person_detected = True

            # Save detection to CSV
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            detections.append([timestamp, class_name, x1, y1, x2, y2, confidence, crop_filename] + keypoint_data)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw Keypoints
            for j, (key_x, key_y) in enumerate(keypoints[i]):
                cv2.circle(frame, (int(key_x), int(key_y)), 3, (0, 0, 255), -1)
                cv2.putText(frame, KEYPOINT_NAMES[j], (int(key_x), int(key_y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Display Class Name & Confidence
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Append detections to CSV file
    if detections:
        df = pd.DataFrame(detections, columns=columns)
        df.to_csv(CSV_FILE, mode="a", header=False, index=False)

    # Play Alert Sound only if a person is detected
    if person_detected:
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play()

    # Show Frame
    cv2.imshow("YOLOv5 Pose Estimation", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

