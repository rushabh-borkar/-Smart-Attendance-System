import os
import urllib.request
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from config import (
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    YUNET_URL,
    SFACE_URL,
    DEFAULT_SIMILARITY_THRESHOLD,
    CONFIDENCE_DETECTION_THRESHOLD
)

class FaceEngine:
    def __init__(self):
        self._ensure_models_exist()
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_PATH,
            config="",
            input_size=(320, 320),
            score_threshold=CONFIDENCE_DETECTION_THRESHOLD,
            nms_threshold=0.3,
            top_k=5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_MODEL_PATH,
            config=""
        )

    def _ensure_models_exist(self):
        """Automatically downloads the ONNX models from official OpenCV repository if missing."""
        os.makedirs(os.path.dirname(YUNET_MODEL_PATH), exist_ok=True)
        if not os.path.exists(YUNET_MODEL_PATH):
            print("Downloading YuNet face detection model...")
            urllib.request.urlretrieve(YUNET_URL, YUNET_MODEL_PATH)
        if not os.path.exists(SFACE_MODEL_PATH):
            print("Downloading SFace face recognition model...")
            urllib.request.urlretrieve(SFACE_URL, SFACE_MODEL_PATH)

    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detects faces in an RGB/BGR image.
        Returns a list of detected face arrays (each has 15 elements: bbox, landmarks, score).
        """
        if image is None or image.size == 0:
            return []
        
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        
        if faces is None or len(faces) == 0:
            return []
        return [f for f in faces]

    def extract_embedding(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """
        Aligns the face and extracts a 128-dimensional embedding vector.
        """
        aligned_face = self.recognizer.alignCrop(image, face)
        embedding = self.recognizer.feature(aligned_face)
        return embedding

    def get_primary_face_embedding(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Used during user registration:
        Checks that exactly one clear face is present, and returns (embedding, face_data, status_message).
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return None, None, "No face detected. Please ensure your face is well-lit and directly facing the camera."
        if len(faces) > 1:
            return None, None, "Multiple faces detected. Please ensure only one person is in the frame."
        
        face = faces[0]
        # Check face size (ensure face is reasonably close)
        w, h = face[2], face[3]
        if w < 60 or h < 60:
            return None, None, "Face is too far away or too small. Please move closer to the camera."

        embedding = self.extract_embedding(image, face)
        return embedding, face, "Face successfully detected and processed!"

    def compare_faces(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two embeddings.
        Typical match threshold: >= 0.363 to 0.40.
        """
        score = self.recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)

    def match_against_known(
        self,
        current_emb: np.ndarray,
        known_users: List[Dict[str, Any]],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Finds the best matching known user from database users list.
        Returns (best_match_user, highest_similarity_score).
        """
        best_user = None
        best_score = -1.0

        for user in known_users:
            stored_emb = user.get("embedding")
            if stored_emb is None:
                continue
            
            # Reshape if necessary
            if stored_emb.ndim == 1:
                stored_emb = stored_emb.reshape(1, -1)
            if current_emb.ndim == 1:
                current_emb = current_emb.reshape(1, -1)

            similarity = self.compare_faces(current_emb, stored_emb)
            if similarity > best_score:
                best_score = similarity
                if similarity >= threshold:
                    best_user = user

        return best_user, max(best_score, 0.0)

    @staticmethod
    def draw_face_hud(
        frame: np.ndarray,
        face: np.ndarray,
        name: str,
        subtitle: str,
        status_color: Tuple[int, int, int] = (0, 255, 0),
        confidence_pct: Optional[float] = None
    ) -> np.ndarray:
        """
        Renders a polished, modern futuristic HUD bounding box and badge tag on the frame.
        status_color: (B, G, R)
        """
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        
        # Ensure coordinates are within image boundaries
        img_h, img_w = frame.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(10, min(w, img_w - x))
        h = max(10, min(h, img_h - y))

        corner_len = int(min(w, h) * 0.25)
        thickness = 2
        corner_thickness = 4

        # Draw semi-transparent rounded-corner bounding box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), status_color, thickness)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Draw stylish corner accents
        # Top-Left
        cv2.line(frame, (x, y), (x + corner_len, y), status_color, corner_thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), status_color, corner_thickness)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), status_color, corner_thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), status_color, corner_thickness)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), status_color, corner_thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), status_color, corner_thickness)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), status_color, corner_thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), status_color, corner_thickness)

        # Draw facial landmarks (eyes, nose, mouth corners)
        landmarks = face[4:14].reshape(5, 2).astype(np.int32)
        for lx, ly in landmarks:
            cv2.circle(frame, (lx, ly), 3, (0, 255, 255), -1)

        # Draw Label Header Badge
        header_text = name
        if confidence_pct is not None:
            header_text += f" ({confidence_pct:.0f}%)"

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        badge_margin = 6
        (text_w, text_h), baseline = cv2.getTextSize(header_text, font, font_scale, 1)

        badge_y1 = max(0, y - text_h - 20)
        badge_y2 = y
        badge_x1 = x
        badge_x2 = x + max(text_w + 20, 120)

        # Background badge for text
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (25, 25, 25), -1)
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), status_color, 1)

        # Put header text
        cv2.putText(frame, header_text, (badge_x1 + badge_margin, badge_y1 + text_h + badge_margin - 2),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Put subtext (Status or ID)
        if subtitle:
            sub_font_scale = 0.42
            sub_y1 = y + h
            sub_y2 = y + h + 22
            cv2.rectangle(frame, (x, sub_y1), (x + text_w + 20, sub_y2), (25, 25, 25), -1)
            cv2.rectangle(frame, (x, sub_y1), (x + text_w + 20, sub_y2), status_color, 1)
            cv2.putText(frame, subtitle, (x + badge_margin, sub_y2 - 6),
                        font, sub_font_scale, status_color, 1, cv2.LINE_AA)

        return frame
