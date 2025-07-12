from ultralytics import YOLO
import cv2
from cap_from_youtube import cap_from_youtube

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")  # or your model path

# Open youtube video
link = "https://www.youtube.com/watch?v=KBsqQez-O4w"
cap = cap_from_youtube(link, "720p")

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference
    results = model(frame,     
                    classes=[2, 3, 5, 7] # tracking car, motorcycle, bus, truck
     ) 

    # Get detections from the first result (assuming one frame)
    boxes = results[0].boxes
    class_names = model.names  # {0: 'car', 1: 'truck', ...}

    for box in boxes:
        cls_id = int(box.cls[0])  # class index
        label = class_names[cls_id]
        conf = float(box.conf[0])  # confidence score

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # [x1, y1, x2, y2]

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Show frame
    cv2.imshow("YOLOv8 Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
