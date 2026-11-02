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

video_file = 'dataset/traffic_sample.mp4'

if st.button("Start Live Detection Stream"):
    cap = cv2.VideoCapture(video_file)
    frame_placeholder = st.empty()
    
    col1, col2, col3 = st.columns(3)
    kpi1 = col1.empty()
    kpi2 = col2.empty()
    kpi3 = col3.empty()

    VEHICLE_CLASSES = [2, 3, 5, 7] # Car, motorcycle, bus, truck

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video stream finished.")
            break

        frame = cv2.resize(frame, (800, 450))
        results = detection_model(frame, stream=True)
        
        vehicle_count = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Traffic Density
        if vehicle_count < 10:
            density = "LOW DENSITY"
            d_color = "green"
        elif vehicle_count < 25:
            density = "MEDIUM DENSITY"
            d_color = "orange"
        else:
            density = "HIGH DENSITY"
            d_color = "red"

        # Risk Prediction
        risk_pred = risk_model.predict(np.array([[vehicle_count, 65, 1]]))[0]
        risk_labels = {0: "LOW RISK", 1: "MODERATE RISK", 2: "HIGH RISK"}

        # Update Metrics UI
        kpi1.metric("Vehicle Count", vehicle_count)
        kpi2.markdown(f"**Density:** :{d_color}[{density}]")
        kpi3.metric("Accident Risk", risk_labels.get(risk_pred, "UNKNOWN"))

        # Render frame on Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB")

    cap.release()