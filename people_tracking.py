import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = "video.mp4"
OUTPUT_PATH = "output_compressed.mp4"
RESIZE_WIDTH = 640   # Change to 480 / 800 if needed

# -----------------------------
# LOAD MODELS
# -----------------------------
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# -----------------------------
# LOAD VIDEO
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate new height while keeping aspect ratio
aspect_ratio = original_height / original_width
RESIZE_HEIGHT = int(RESIZE_WIDTH * aspect_ratio)

# Output Video Writer (compressed size)
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (RESIZE_WIDTH, RESIZE_HEIGHT)
)

previous_positions = {}

# -----------------------------
# MAIN LOOP
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 COMPRESS FRAME (Resize)
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    detections = []
    results = model(frame)

    # -----------------------------
    # PERSON DETECTION
    # -----------------------------
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # Person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                detections.append(
                    ([x1, y1, x2 - x1, y2 - y1], conf, "person")
                )

    # -----------------------------
    # TRACKING
    # -----------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        w = r - l
        h = b - t
        x_center = l + w // 2

        # -----------------------------
        # ACTIVITY CLASSIFICATION
        # -----------------------------
        activity = "Standing"

        if track_id in previous_positions:
            prev_x = previous_positions[track_id]
            movement = abs(x_center - prev_x)

            if movement > 8:   # smaller threshold for small frame
                activity = "Walking"

        # Sitting detection
        if h < 120:   # Adjust based on resolution
            activity = "Sitting"

        previous_positions[track_id] = x_center

        # -----------------------------
        # DRAW RESULTS
        # -----------------------------
        color = (0, 255, 0)

        cv2.rectangle(frame, (l, t), (r, b), color, 2)

        cv2.putText(
            frame,
            f"ID {track_id} : {activity}",
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(frame)
    cv2.imshow("Compressed People Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# RELEASE
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()