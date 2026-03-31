# YOLO-Based Smart Surveillance System with Suspicious Clip Recording
project submission as per course requirements

I have made an AI-based surveillance system that detects, tracks and analyzes human behaviour based on their movement patterns in real time using YOLO. It identifies suspicious activities like loitering and automatically records video clips for the same. 

core features: 
1. detect and track people in real time using YOLOv8
2. analyze movement patterns
3. detect suspicious behvaiour (loitering)
4. highlight risk zones in real time (ROI monitoring)
5. multi object tracking
6. motion heatmap generation
7. automatic video recording
8. command line execution

Tech stack:
1. python
2. openCV
3. ultralytics YOLOv8
4. NumPy

Installation and setup:

requirements:
1. ultralytics
2. opencv-python
3. numpy

Step 1: Clone Repository
git clone https://github.com/your-username/smart-surveillance-yolo.git
cd smart-surveillance-yolo
Step 2: Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Step 3: Install Dependencies
pip install -r requirements.txt

#WEBCAM ACCESS IS REQUIRED

Run the Project: python main.py

Saved suspicious clips will be stored in the output/ folder.

Future improvments:
1. real time notifications/alerts
2. cloud storage
3. face recognition
