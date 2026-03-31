import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

os.makedirs("output", exist_ok=True)

model = YOLO("yolov8n.pt")  

cap = cv2.VideoCapture(0)

# Tracking storage
object_times = {}

# Heatmap
heatmap = None

# Recording
recording = False
out = None
record_start = 0

# Settings
LOITER_TIME = 10        
RECORD_DURATION = 5     

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Initialize heatmap
    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)

    # YOLO Detection
    results = model(frame, verbose=False)

    alert = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Centroid
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Simple grid-based ID
                obj_id = (cx // 50, cy // 50)

                # Track time
                if obj_id not in object_times:
                    object_times[obj_id] = time.time()

                duration = time.time() - object_times[obj_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Update heatmap
                heatmap[y1:y2, x1:x2] += 1

                # Loitering detection
                if duration > LOITER_TIME:
                    alert = True
                    cv2.putText(frame, "LOITERING ALERT!",
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

    if alert and not recording:
        filename = f"output/suspicious_{int(time.time())}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
        recording = True
        record_start = time.time()
        print(f"[INFO] Recording started: {filename}")

    if recording:
        out.write(frame)

        # Stop after fixed duration
        if time.time() - record_start > RECORD_DURATION:
            recording = False
            out.release()
            print("[INFO] Recording saved")

    cv2.imshow("Smart Surveillance System", overlay)

    try:
        if cv2.getWindowProperty("Smart Surveillance System", cv2.WND_PROP_VISIBLE) < 1:
            break

    except:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()