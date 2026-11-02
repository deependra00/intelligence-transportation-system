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

        # Modern Card Styling for Metrics
        with col1:
            st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border-left: 5px solid #1f77b4; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                    <p style="color: #31333F; font-size: 14px; margin: 0; font-weight: bold;">🚗 Vehicle Count</p>
                    <h2 style="color: #1f77b4; margin: 5px 0 0 0;">{vehicle_count}</h2>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border-left: 5px solid #ffa500; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                    <p style="color: #31333F; font-size: 14px; margin: 0; font-weight: bold;">📊 Traffic Density</p>
                    <h3 style="color: {d_color}; margin: 5px 0 0 0;">{density}</h3>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border-left: 5px solid #ff4b4b; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                    <p style="color: #31333F; font-size: 14px; margin: 0; font-weight: bold;">⚠️ Accident Risk</p>
                    <h3 style="color: #ff4b4b; margin: 5px 0 0 0;">{risk_labels.get(risk_pred, "UNKNOWN")}</h3>
                </div>
            """, unsafe_allow_html=True)

        # Render frame on Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB")

    cap.release()