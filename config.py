import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
DATABASE_PATH = DATA_DIR / "attendance.db"

# Create directories if not existing
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
YUNET_MODEL_PATH = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")

# URLs to download models if missing
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# AI Recognition Thresholds
# SFace Cosine similarity threshold: higher means stricter match. Standard is 0.34 - 0.37 (optimal 0.35)
DEFAULT_SIMILARITY_THRESHOLD = 0.35
CONFIDENCE_DETECTION_THRESHOLD = 0.65

# Attendance Business Rules
DEFAULT_LATE_CUTOFF_TIME = "09:30:00"  # Format: HH:MM:SS
ATTENDANCE_COOLDOWN_SECONDS = 30       # Minimum time before duplicate scan notification
