import streamlit as st
import cv2
import joblib
import numpy as np
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="ITS & IIoT Intelligence Dashboard",
    page_icon="🚗",
    layout="wide"
)

# --- CUSTOM CSS STYLING FOR ATTRACTIVE UI ---
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Title Styling */
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        padding: 15px;
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #334155;
    }

    /* Custom Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.4);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">🚗 Intelligent Transportation System (ITS) - IIoT Dashboard</div>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    detection_model = YOLO('yolov8n.pt')
    risk_model = joblib.load('models/risk_predictor.pkl')
    return detection_model, risk_model

detection_model, risk_model = load_models()

video_file = 'dataset/traffic_sample.mp4'

# Sidebar Controls
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.markdown("---")
start_stream = st.sidebar.button("🚀 Start Live Detection Stream")

# Layout Containers
col_video, col_metrics = st.columns([2, 1])

with col_video:
    st.markdown("### 📹 Real-Time Video Feed")
    frame_placeholder = st.empty()

with col_metrics:
    st.markdown("### 📊 Live Analytics")
    metric1_placeholder = st.empty()
    metric2_placeholder = st.empty()
    metric3_placeholder = st.empty()

if start_stream:
    cap = cv2.VideoCapture(video_file)
    VEHICLE_CLASSES = [2, 3, 5, 7] # Car, motorcycle, bus, truck

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video stream finished.")
            break

        frame = cv2.resize(frame, (750, 420))
        results = detection_model(frame, stream=True)
        
        vehicle_count = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Traffic Density Logic
        if vehicle_count < 10:
            density = "LOW DENSITY"
            d_color = "#10b981" # Green
        elif vehicle_count < 25:
            density = "MEDIUM DENSITY"
            d_color = "#f59e0b" # Orange
        else:
            density = "HIGH DENSITY"
            d_color = "#ef4444" # Red

        # Risk Prediction
        risk_pred = risk_model.predict(np.array([[vehicle_count, 65, 1]]))[0]
        risk_labels = {0: "LOW RISK", 1: "MODERATE RISK", 2: "HIGH RISK"}
        risk_colors = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444"}

        # Modern Glassmorphism Metric Cards Update
        metric1_placeholder.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); padding: 16px; border-radius: 12px; text-align: center; border-left: 6px solid #3b82f6; box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom: 12px;">
                <p style="color: #94a3b8; font-size: 13px; margin: 0; font-weight: 600; text-transform: uppercase;">Vehicle Count</p>
                <h2 style="color: #60a5fa; margin: 5px 0 0 0; font-size: 28px;">{vehicle_count}</h2>
            </div>
        """, unsafe_allow_html=True)

        metric2_placeholder.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); padding: 16px; border-radius: 12px; text-align: center; border-left: 6px solid {d_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom: 12px;">
                <p style="color: #94a3b8; font-size: 13px; margin: 0; font-weight: 600; text-transform: uppercase;">Traffic Density</p>
                <h3 style="color: {d_color}; margin: 5px 0 0 0; font-size: 20px;">{density}</h3>
            </div>
        """, unsafe_allow_html=True)

        metric3_placeholder.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); padding: 16px; border-radius: 12px; text-align: center; border-left: 6px solid {risk_colors.get(risk_pred, '#ef4444')}; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                <p style="color: #94a3b8; font-size: 13px; margin: 0; font-weight: 600; text-transform: uppercase;">Accident Risk</p>
                <h3 style="color: {risk_colors.get(risk_pred, '#ef4444')}; margin: 5px 0 0 0; font-size: 20px;">{risk_labels.get(risk_pred, "UNKNOWN")}</h3>
            </div>
        """, unsafe_allow_html=True)

        # Render frame on Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    cap.release()
else:
    # Initial placeholder cards when stream is not started
    metric1_placeholder.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 16px; border-radius: 12px; text-align: center; border-left: 6px solid #3b82f6; box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom: 12px;">
            <p style="color: #94a3b8; font-size: 13px; margin: 0; font-weight: 600; text-transform: uppercase;">Vehicle Count</p>
            <h2 style="color: #60a5fa; margin: 5px 0 0 0; font-size: 28px;">0</h2>
        </div>
    """, unsafe_allow_html=True)
    metric2_placeholder.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 16px; border-radius: 12px; text-align: center; border-left: 6px solid #10b981; box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom: 12px;">
            <p style="color: #94a3b8; font-size: 13px; margin: 0; font-weight: 600; text-transform: uppercase;">Traffic Density</p>
            <h3 style="color: #10b981; margin: 5px 0 0 0; font-size: 20px;">IDLE</h3>
        </div>
    """, unsafe_allow_html=True)
    metric3_placeholder.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 16px; border-radius: 12px; text-align: center; border-left: 6px solid #10b981; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
            <p style="color: #94a3b8; font-size: 13px; margin: 0; font-weight: 600; text-transform: uppercase;">Accident Risk</p>
            <h3 style="color: #10b981; margin: 5px 0 0 0; font-size: 20px;">LOW RISK</h3>
        </div>
    """, unsafe_allow_html=True)