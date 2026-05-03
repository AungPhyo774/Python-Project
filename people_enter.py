import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime


# Load YOLO model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# Open video file
cap = cv2.VideoCapture("video20.mp4")

# Logs and counters
log_data = []
entry_count = 0   # Right → Left (Enter)
exit_count = 0    # Left → Right (Exit)

# Store previous x positions to detect crossing
track_prev_x = {}

line_x = 500  # vertical line

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (660, 540))

    # YOLO detection
    results = model(frame)[0]
    detections = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        if int(class_id) == 0:  # only person
            detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        # Draw bounding box
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Get previous x position
        prev_x = track_prev_x.get(track_id, cx)
        
        # Left → Right = Exit
        if prev_x < line_x <= cx:
            exit_count += 1
            log_data.append({"ID": track_id, "Time": datetime.now(), "Event": "Exit"})

        # Right → Left = Enter
        elif prev_x > line_x >= cx:
            entry_count += 1
            log_data.append({"ID": track_id, "Time": datetime.now(), "Event": "Enter"})

        # Update previous x
        track_prev_x[track_id] = cx

    # Draw vertical line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)

    # Show counts
    cv2.putText(frame, f"Enter: {entry_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"Exit: {exit_count}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Personal Tracking System", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release and save logs
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(log_data)
df.to_csv("tracking_log.csv", index=False)
print("Tracking log saved to tracking_log.csv")