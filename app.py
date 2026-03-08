import streamlit as st
from PIL import Image
import cv2
import os
import datetime

from utils.detection import detect_objects
from utils.analytics import generate_object_chart
from utils.report import classify_scene, generate_ai_report
from utils.live_cctv import process_live_frame

if "run_image" not in st.session_state:
    st.session_state.run_image = False

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

st.title("AI Image Intelligence & Live CCTV Monitoring System")

mode = st.radio(
    "Select Mode",
    ["Image Analysis", "Live CCTV Monitoring"]
)

if mode != "Image Analysis":
    st.session_state.run_image = False

if mode == "Image Analysis":

    st.subheader("Image Analysis Dashboard")

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        if st.button("Run Object Detection"):
            st.session_state.run_image = True

        if st.session_state.run_image:

            with st.spinner("Detecting objects..."):
                annotated_image, detected_objects = detect_objects(image)

            st.subheader("Detected Image")
            st.image(annotated_image, use_column_width=True)

            if detected_objects:

                # Detection Results
                st.subheader("Detection Results")
                for obj in detected_objects:
                    st.write(f"• {obj['label']} (confidence: {obj['confidence']})")

                # Confidence Insights
                st.subheader("Confidence Insights")
                confidences = [obj["confidence"] for obj in detected_objects]
                avg_conf = sum(confidences) / len(confidences)
                max_conf = max(detected_objects, key=lambda x: x["confidence"])
                st.write(f"Average Confidence: {avg_conf:.2f}")
                st.write(
                    f"Highest Confidence Object: "
                    f"{max_conf['label']} ({max_conf['confidence']})"
                )

                # Analytics (Per Image)
                st.subheader("Analytics")
                chart = generate_object_chart(detected_objects)
                if chart:
                    st.pyplot(chart)

                # Scene Classification
                scene = classify_scene(detected_objects)
                st.subheader("ML Scene Classification")
                st.success(f"Predicted Scene: {scene}")

                # AI Report
                st.subheader("GenAI Automated Report")
                ai_report = generate_ai_report(detected_objects, scene)
                st.write(ai_report)

                # Download Report
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=ai_report,
                    file_name="image_analysis_report.txt",
                    mime="text/plain"
                )

            else:
                st.warning("No objects detected in the image.")

# ======================================================
# MODE 2: LIVE CCTV MONITORING
# ======================================================
if mode == "Live CCTV Monitoring":

    st.subheader("Live CCTV / Webcam Monitoring")
    st.warning("This module simulates CCTV using a webcam")

    # Threshold Configuration
    st.subheader("Alert Configuration")
    person_threshold = st.slider(
        "Suspicious Activity Threshold (Number of people)",
        min_value=1,
        max_value=10,
        value=3
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start CCTV Camera"):
            st.session_state.camera_on = True
    with col2:
        if st.button("Stop CCTV Camera"):
            st.session_state.camera_on = False

    FRAME_WINDOW = st.image([])
    count_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0)

        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Unable to access camera")
                break

            annotated_frame, person_count, suspicious = process_live_frame(
                frame,
                person_threshold=person_threshold
            )

            FRAME_WINDOW.image(annotated_frame)
            count_placeholder.info(f"People detected: {person_count}")

            if suspicious:
                status_placeholder.error("⚠️ Suspicious Activity Detected")

                # Save snapshot
                os.makedirs("alerts", exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                cv2.imwrite(f"alerts/alert_{timestamp}.jpg", frame)

                # Log alert (avoid duplicates)
                if len(st.session_state.alert_history) == 0 or \
                   st.session_state.alert_history[-1]["time"] != timestamp:
                    st.session_state.alert_history.append({
                        "time": timestamp,
                        "people_count": person_count
                    })
            else:
                status_placeholder.success("Normal Activity")

        cap.release()

    # Alert History
    if st.session_state.alert_history:
        st.subheader("🚨 Alert History")
        st.table(st.session_state.alert_history)

        st.download_button(
            label="⬇️ Download Alert Log",
            data=str(st.session_state.alert_history),
            file_name="cctv_alert_log.txt",
            mime="text/plain"
        )