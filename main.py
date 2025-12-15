import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # lightweight model (good for drones)

# Open video (0 for webcam, or video file)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    center_x = width // 2

    # YOLO inference
    results = model(frame, conf=0.5)

    direction = "Forward"

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_center = (x1 + x2) // 2

            # Decision logic
            if obj_center < center_x - 50:
                direction = "Move Right"
            elif obj_center > center_x + 50:
                direction = "Move Left"
            else:
                direction = "Move Up"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show direction
    cv2.putText(frame, f"Decision: {direction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Obstacle Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
