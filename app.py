import streamlit as st
import cv2
import joblib
import numpy as np
from ultralytics import YOLO
import tempfile

st.title("🚗 Intelligent Transportation System (ITS) - IIoT Dashboard")

@st.cache_resource
def load_models():
    det_model = YOLO('yolov8n.pt')
    r_model = joblib.load('models/risk_predictor.pkl')
    return det_model, r_model

detection_model, risk_model = load_models()
VEHICLE_CLASSES = [2, 3, 5, 7]

uploaded_file = st.file_uploader("Upload a Traffic Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    stframe = st.empty()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 6 != 0:
            continue
        
        frame = cv2.resize(frame, (480, 270))
        results = detection_model(frame, stream=True, imgsz=320, conf=0.4)
        vehicle_count = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in VEHICLE_CLASSES:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        density = "LOW" if vehicle_count < 10 else ("MEDIUM" if vehicle_count <= 25 else "HIGH")
        risk_pred = risk_model.predict(np.array([[vehicle_count, 65, 1]]))[0]
        risk_text = ["LOW RISK", "MODERATE RISK", "HIGH RISK"][risk_pred]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)
        
        st.sidebar.markdown(f"**Vehicle Count:** {vehicle_count}")
        st.sidebar.markdown(f"**Density:** {density}")
        st.sidebar.markdown(f"**Accident Risk:** {risk_text}")