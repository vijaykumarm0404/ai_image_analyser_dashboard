import cv2
from utils.detection import detect_objects

def process_live_frame(frame, person_threshold=3):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    annotated_frame, detected_objects = detect_objects(frame_rgb)

    person_count = sum(
        1 for obj in detected_objects if obj["label"] == "person"
    )

    suspicious = person_count >= person_threshold

    return annotated_frame, person_count, suspicious