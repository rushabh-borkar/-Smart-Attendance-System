# 🛡️ AI-Based Smart Attendance Monitoring with Face Recognition

A state-of-the-art, automated biometric attendance monitoring system powered by deep learning facial detection and recognition algorithms (OpenCV YuNet + SFace ONNX deep embeddings).

---

## ✨ Features

- **🚀 Real-Time Face Detection & Recognition**: Powered by OpenCV's native deep neural network models (`YuNet` detector + `SFace` 128-d cosine embedding recognizer). No heavy C++ CMake or dlib build tool dependencies required!
- **📸 Modern Web Dashboard**: Interactive Streamlit portal featuring:
  - **Live Camera Scanner**: Instant webcam snapshot or video input with immediate visual & text feedback.
  - **Student / Employee Enrollment**: Enroll new individuals with live single-face validation, auto-cropping, and embedding generation.
  - **Analytics & Reporting**: Daily attendance rates, punctuality KPIs (On-time vs. Late arrivals), and absent tracking.
  - **Export Capabilities**: One-click download of attendance logs to **Excel (.xlsx)** and **CSV (.csv)**.
  - **Directory Management**: Search, view enrolled profiles with photos, or remove accounts.
  - **System Settings**: Adjust late arrival cutoff time and recognition sensitivity dynamically.
- **🖥️ Dedicated Desktop Kiosk (`kiosk_monitor.py`)**: High-speed 60 FPS OpenCV camera monitoring window with futuristic HUD bounding boxes, audio chime on check-in, and auto-logging.
- **💾 Robust Database**: Embedded SQLite database tracking users, 128-d embeddings, check-in timestamps, confidence scores, and status flags.

---

## 🏗️ System Architecture

```
AI BASED ATTENDANCE SYSTEM/
│
├── config.py             # System paths, AI thresholds, late cutoff rules
├── database.py           # SQLite database layer (users, attendance logs, settings)
├── face_engine.py        # OpenCV YuNet + SFace face detection, alignment & matching
├── app.py                # Modern Streamlit Web Application Dashboard
├── kiosk_monitor.py      # Standalone high-FPS OpenCV desktop kiosk
├── requirements.txt      # Python dependencies
│
├── models/               # Pretrained deep learning ONNX models
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
└── data/
    ├── attendance.db     # SQLite database
    └── faces/            # Stored reference photos
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
Ensure Python 3.10+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Seed College Demo Dataset & Historical Logs
To populate realistic student/faculty profiles and 14 days of past attendance logs:
```bash
python seed_demo_data.py
```

### 3. Launch the Web Application
To start the web dashboard:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. (Optional) Batch Import Student Photos
Drop student images into the `batch_students/` folder named `RollNo_FullName_Department.jpg` and run:
```bash
python import_folder.py
```

### 5. (Optional) Run the High-Speed Desktop Kiosk
For a dedicated entrance kiosk with audio feedback:
```bash
python kiosk_monitor.py
```
*Tip: To use an external webcam or RTSP camera, pass the camera index or stream URL: `python kiosk_monitor.py 1` or `python kiosk_monitor.py rtsp://user:pass@ip:port/stream`*

---

## 📋 How It Works

1. **Enrollment**: 
   - Navigate to the **"Register New Person"** tab.
   - Enter Student/Employee ID, Full Name, and Department.
   - Take a picture or upload a clear front-facing portrait. The AI validates face presence and saves the 128-d feature vector.
2. **Attendance Marking**:
   - Navigate to **"Live Attendance"** (or run `kiosk_monitor.py`).
   - The AI identifies the individual, checks if attendance has already been logged today, verifies punctuality (against the configured late cutoff time), and logs the event with the exact timestamp.
3. **Audit & Reports**:
   - Access **"Attendance Logs & Analytics"** to filter by date range or department, inspect absentees, and export Excel/CSV reports.
