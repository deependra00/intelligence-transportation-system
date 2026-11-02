An advanced multi-modal artificial intelligence system designed to optimize traffic surveillance, detect vehicles in real-time, estimate roadway density levels, and predict accident hazard risks using deep learning and machine learning models. Built as a collaborative MCA project (Group 4) at Invertis University.

Tech Stack & Libraries
Programming Language: Python 3.10+

Computer Vision & Object Detection: Ultralytics YOLOv8 (yolov8n.pt), OpenCV (cv2)

Machine Learning & Analytics: Scikit-Learn, Pandas, NumPy, Joblib (risk_predictor.pkl)

Web Presentation & Dashboard: Streamlit Community Cloud

Version Control & Hosting: Git, GitHub, Render, Vercel

Project Directory Structure
Plaintext
intelligence-transportation-system/
│
├── models/
│   └── risk_predictor.pkl      # Serialized Scikit-Learn risk classification model
│
├── .gitignore                  # Excludes large datasets and cache files
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # Repository documentation
└── traffic_sample.mp4          # Sample traffic video clip for testing

Key Features & Modules
Real-Time Vehicle Detection: Utilizes YOLOv8 nano pre-trained weights to detect cars, motorcycles, buses, and trucks while ignoring background noise.

Dynamic Traffic Density Estimation: Automatically categorizes traffic volume per frame into LOW (<10 vehicles), MEDIUM (10–25 vehicles), and HIGH (>25 vehicles).

Accident Risk Prediction: Feeds real-time counts and environmental parameters into a Random Forest classifier to assess roadway hazard severity (LOW RISK, MODERATE RISK, HIGH RISK).

Cloud-Optimized Web App: Streamlit dashboard configured with frame-skipping and memory management (gc.collect()) to run smoothly on resource-constrained cloud servers.

Installation & Local Execution
Clone the Repository:

Bash
git clone https://github.com/deependra00/intelligence-transportation-system.git
cd intelligence-transportation-system
Install Dependencies:

Bash
pip install -r requirements.txt
Run the Streamlit Dashboard:

Bash
streamlit run app.py
🌐 Live Demo
You can access the live web application hosted on Streamlit Cloud:




