import torch
import cv2
import numpy as np
import pandas as pd
import pygame
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
import json

# Initialize Pygame for Sound Alerts
pygame.mixer.init()
WELCOME_SOUND = "welcome.mp3"  # Ensure this file exists

# Load YOLOv5s Model (Small version for better speed)
model = YOLO("yolov5s.pt")  

# COCO Class Labels
CLASS_NAMES = model.names  # YOLO automatically loads class names

# Define Non-Maximum Suppression (NMS) Parameters
NMS_THRESHOLD = 0.3  
CONFIDENCE_THRESHOLD = 0.6  

# Setup output directories and files
CSV_DIR = "detections"
Path(CSV_DIR).mkdir(exist_ok=True)  # Create folder if it doesn't exist

# Create CSV and JSON file names with timestamp
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
CSV_FILE = f"{CSV_DIR}/detections_{timestamp_str}.csv"
JSON_FILE = f"{CSV_DIR}/detections_{timestamp_str}.json"

# CSV columns
columns = ["Timestamp", "Class", "X1", "Y1", "X2", "Y2", "Confidence", "Image_Path"]

# Create CSV with headers
pd.DataFrame(columns=columns).to_csv(CSV_FILE, index=False)

# Initialize list for JSON detections
json_results = []

# Open video file (or change to 0 for webcam)
cap = cv2.VideoCapture(0)

frame_id = 0
last_welcome_time = datetime.now() - timedelta(seconds=3)  # To track last welcome message

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    detections = []  # For storing detections in the current frame
    person_detected = False  # To trigger welcome sound if a person is detected
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()   # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()   # Class IDs

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            else:
                indices = [indices]

            for i in indices:
                x1, y1, x2, y2 = map(int, boxes[i])
                confidence = float(scores[i])
                label = int(class_ids[i])
                
                # Process only if the detection is a person (class ID 0)
                if label != 0:
                    continue

                class_name = CLASS_NAMES.get(label, "Unknown")
                
                # Create a subfolder for the class "person"
                class_folder = Path(CSV_DIR) / class_name
                class_folder.mkdir(exist_ok=True)
                
                # Crop the detected person
                object_crop = frame[y1:y2, x1:x2]

                # Add timestamp to the cropped image
                timestamp_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                color = (255, 255, 255)  # White text
                bg_color = (0, 0, 0)  # Black background for contrast

                # Get text size
                text_size = cv2.getTextSize(timestamp_text, font, font_scale, font_thickness)[0]
                text_x = 5  # Left margin
                text_y = text_size[1] + 5  # Top margin

                # Draw rectangle background for text
                cv2.rectangle(object_crop, (text_x - 2, text_y - text_size[1] - 2),
                              (text_x + text_size[0] + 2, text_y + 2), bg_color, -1)

                # Put timestamp text on image
                cv2.putText(object_crop, timestamp_text, (text_x, text_y), font,
                            font_scale, color, font_thickness, cv2.LINE_AA)

                # Save the cropped image with timestamp
                crop_filename = f"{class_folder}/crop_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(crop_filename, object_crop)

                person_detected = True

                # Save detection to CSV and JSON
                current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                detection_data = {
                    "Timestamp": current_timestamp,
                    "Class": class_name,
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                    "Confidence": confidence,
                    "Image_Path": str(crop_filename)
                }
                detections.append(detection_data)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save frame's detection data (only persons) to JSON results
    frame_result = {
        "frame_id": frame_id,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "detections": detections
    }
    json_results.append(frame_result)

    # Append detections to CSV file if any were detected
    if detections:
        df = pd.DataFrame(detections, columns=columns)
        df.to_csv(CSV_FILE, mode="a", header=False, index=False)

    # Play welcome sound if a person is detected and 3 seconds have passed
    current_time = datetime.now()
    if person_detected and (current_time - last_welcome_time).total_seconds() > 3:
        pygame.mixer.music.load(WELCOME_SOUND)
        pygame.mixer.music.play()
        last_welcome_time = current_time  # Update the last played time

    cv2.imshow("YOLOv5s Person Detection", frame)
    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Write JSON results to file at the end
with open(JSON_FILE, "w") as json_file:
    json.dump(json_results, json_file, indent=4)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print(f"Detections exported to CSV: {CSV_FILE}")
print(f"Detections exported to JSON: {JSON_FILE}")

