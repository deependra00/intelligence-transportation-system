import streamlit as st
import cv2
import joblib
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="ITS Dashboard", layout="wide")

st.title("🚗 Intelligent Transportation System (ITS) - IIoT Dashboard")

@st.cache_resource
def load_models():
    detection_model = YOLO('yolov8n.pt')
    risk_model = joblib.load('models/risk_predictor.pkl')
    return detection_model, risk_model

detection_model, risk_model = load_models()

VEHICLE_CLASSES = [2, 3, 5, 7]

video_file = 'dataset/traffic_sample.mp4'

# Streamlit Layout
st.video('dataset/traffic_sample.mp4')

frame_placeholder = st.empty()
kpi_placeholder = st.empty()

if run_stream:
    cap = cv2.VideoCapture(video_file)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video stream ended or failed to grab frame.")
            break

        frame = cv2.resize(frame, (960, 540))

        # YOLO Vehicle Detection
        results = detection_model(frame, stream=True)
        
        vehicle_count = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Traffic Density Classification
        if vehicle_count < 10:
            density = "LOW DENSITY"
            density_color = "green"
        elif vehicle_count < 25:
            density = "MEDIUM DENSITY"
            density_color = "orange"
        else:
            density = "HIGH DENSITY"
            density_color = "red"

        # Accident Risk Prediction
        dummy_speed = 65
        dummy_weather = 1  
        risk_input = np.array([[vehicle_count, dummy_speed, dummy_weather]])
        risk_pred = risk_model.predict(risk_input)[0]

        risk_labels = {0: "LOW RISK", 1: "MODERATE RISK", 2: "HIGH RISK"}
        risk_text = risk_labels.get(risk_pred, "UNKNOWN")

        # Display Metrics on Sidebar/KPI section
        with kpi_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Vehicle Count", vehicle_count)
            col2.markdown(f"**Density:** :{density_color}[{density}]")
            col3.metric("Accident Risk", risk_text)

        # Convert OpenCV BGR frame to RGB for Streamlit display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB")

    cap.release()