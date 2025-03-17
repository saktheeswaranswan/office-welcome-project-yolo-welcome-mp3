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

# Load YOLOv5m Model (Medium version for better accuracy)
model = YOLO("yolov5m.pt")  

# COCO Class Labels
CLASS_NAMES = model.names  # YOLO automatically loads class names

# Define Non-Maximum Suppression (NMS) Parameters
NMS_THRESHOLD = 0.3  
CONFIDENCE_THRESHOLD = 0.6  

# CSV File Setup
CSV_DIR = "detections"
Path(CSV_DIR).mkdir(exist_ok=True)  # Create folder if it doesn't exist
CSV_FILE = f"{CSV_DIR}/detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
columns = ["Timestamp", "Class", "X1", "Y1", "X2", "Y2", "Confidence", "Image_Path"]

# Create CSV with headers
pd.DataFrame(columns=columns).to_csv(CSV_FILE, index=False)

# Open Video File
cap = cv2.VideoCapture('fgfgg.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    detections = []
    person_detected = False  # Track if at least one person is detected
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Extract confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        if len(indices) > 0:  
            if isinstance(indices, np.ndarray):  # Multiple detections
                indices = indices.flatten()
            else:  # Single detection
                indices = [indices]

            for i in indices:
                x1, y1, x2, y2 = map(int, boxes[i])
                confidence = float(scores[i])
                label = int(class_ids[i])
                class_name = CLASS_NAMES.get(label, "Unknown")  # Get class name

                # Create subfolder for each class
                class_folder = Path(CSV_DIR) / class_name
                class_folder.mkdir(exist_ok=True)

                # Save cropped image for detected object
                crop_filename = f"{class_folder}/crop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                object_crop = frame[y1:y2, x1:x2]
                cv2.imwrite(crop_filename, object_crop)

                # If a person is detected, trigger alert
                if label == 0:
                    person_detected = True

                # Save detection to CSV with timestamp and image path
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                detections.append([timestamp, class_name, x1, y1, x2, y2, confidence, crop_filename])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display class name & confidence
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
    cv2.imshow("YOLOv5m Object Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

