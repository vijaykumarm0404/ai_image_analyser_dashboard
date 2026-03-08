from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 pretrained model (lightweight & fast)
model = YOLO("yolov8n.pt")

def detect_objects(image_pil):
    """
    Takes PIL image
    Returns:
    - annotated image (numpy array)
    - detected objects list
    """

    # Convert PIL image to OpenCV format
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = model(image_bgr)

    detected_objects = []

    # Loop through detections
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])

            detected_objects.append({
                "label": label,
                "confidence": round(confidence, 2)
            })

    # Get annotated image
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return annotated_image, detected_objects
