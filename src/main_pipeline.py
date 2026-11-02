import cv2
import joblib
import numpy as np
from ultralytics import YOLO

print("Loading Models...")
detection_model = YOLO('yolov8n.pt')
risk_model = joblib.load('models/risk_predictor.pkl')

# Vehicle classes mapping in COCO dataset (2: Car, 3: Motorcycle, 5: Bus, 7: Truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

# Video source: 0 for webcam or video file path e.g., 'dataset/traffic_sample.mp4'
cap = cv2.VideoCapture('dataset/traffic_sample.mp4')

print("Starting Live Intelligent Transportation System Stream... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
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
        density_color = (0, 255, 0)
    elif vehicle_count < 25:
        density = "MEDIUM DENSITY"
        density_color = (0, 255, 255)
    else:
        density = "HIGH DENSITY"
        density_color = (0, 0, 255)

    # Accident Risk Prediction Inputs
    dummy_speed = 65
    dummy_weather = 1  # 0: Clear, 1: Rain, 2: Fog
    
    risk_input = np.array([[vehicle_count, dummy_speed, dummy_weather]])
    risk_pred = risk_model.predict(risk_input)[0]

    risk_labels = {0: "LOW RISK", 1: "MODERATE RISK", 2: "HIGH RISK"}
    risk_text = risk_labels.get(risk_pred, "UNKNOWN")

    # UI text Overlays on Screen
    cv2.putText(frame, f"Vehicles Count: {vehicle_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Density: {density}", (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, density_color, 2)
    cv2.putText(frame, f"Accident Risk: {risk_text}", (20, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if risk_pred == 2 else (0, 255, 0), 2)

    # Display Window
    cv2.imshow("ITS - ITsT Dashboard", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()