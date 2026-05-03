import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


model = YOLO("yolov8n.pt")

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0) 


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: resize for faster processing
    frame = cv2.resize(frame, (660, 640))

    # YOLO person detection
    results = model(frame)[0]
    detections = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        if int(class_id) == 0:  # only person
            # DeepSORT format: [x, y, width, height], score, label
            detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()

        # Draw bounding box
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)

        # Draw ID
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw center point
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Show frame
    cv2.imshow("People Tracking with ID", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()