import cv2
import time
import datetime
import sys
import winsound
from typing import Dict

import config
import database
import face_engine

def run_kiosk(camera_source=0):
    print("=" * 60)
    print("🚀 AI SMART ATTENDANCE KIOSK (REAL-TIME WEBCAM MONITOR)")
    print("=" * 60)
    print("Press 'q' or 'ESC' to exit.")
    print("Press 'r' to reload enrolled users from database.")
    print("-" * 60)

    # Initialize Engine & Database
    database.init_db()
    engine = face_engine.FaceEngine()

    # Load users
    known_users = database.get_all_users()
    print(f"Loaded {len(known_users)} enrolled profiles from database.")

    # Open Camera
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera source {camera_source}")
        return

    # Set camera resolution (1280x720 or 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Cooldown tracker to avoid re-announcing every frame
    last_notified: Dict[int, float] = {}
    last_reload_time = time.time()

    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            time.sleep(0.1)
            continue

        # Flip horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_frame_time) if prev_frame_time > 0 else 30.0
        prev_frame_time = current_time

        # Periodically reload users from database every 30 seconds
        if current_time - last_reload_time > 30:
            known_users = database.get_all_users()
            last_reload_time = current_time

        # Read similarity threshold from settings
        sim_threshold = float(database.get_setting("similarity_threshold", str(config.DEFAULT_SIMILARITY_THRESHOLD)))

        # Detect faces in frame
        faces = engine.detect_faces(frame)

        for face in faces:
            embedding = engine.extract_embedding(frame, face)
            matched_user, score = engine.match_against_known(embedding, known_users, threshold=sim_threshold)

            if matched_user:
                user_id = matched_user["id"]
                user_name = matched_user["full_name"]
                user_code = matched_user["user_code"]
                confidence_pct = min(100.0, (score / 0.70) * 100.0)

                # Check cooldown
                now_ts = time.time()
                is_cooldown_active = (user_id in last_notified) and (now_ts - last_notified[user_id] < config.ATTENDANCE_COOLDOWN_SECONDS)

                # Check if marked today
                today_rec = database.check_attendance_today(user_id)

                if not today_rec:
                    # New check-in for today!
                    success, msg, rec = database.record_attendance(user_id, confidence=score)
                    if success:
                        last_notified[user_id] = now_ts
                        # Audio Beep
                        try:
                            winsound.Beep(1400, 150)
                        except Exception:
                            pass
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ {user_name} ({user_code}) -> {rec.get('status')} marked!")

                    status_color = (0, 255, 0) # Green
                    subtitle = f"MARKED: {rec.get('status', 'Present').upper()}"
                else:
                    # Already marked today
                    status_color = (0, 165, 255) # Orange warning
                    subtitle = f"ALREADY MARKED @ {today_rec['time']}"

                frame = face_engine.FaceEngine.draw_face_hud(
                    frame, face, user_name, subtitle, status_color=status_color, confidence_pct=confidence_pct
                )
            else:
                # Unknown Person
                frame = face_engine.FaceEngine.draw_face_hud(
                    frame, face, "Unknown", "NOT REGISTERED", status_color=(0, 0, 255), confidence_pct=None
                )

        # Draw Top Status Banner
        today_records = database.get_today_attendance()
        banner_h = 45
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Info texts
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_text = f"AI ATTENDANCE KIOSK | {date_str} | FPS: {fps:.0f} | Present Today: {len(today_records)}"
        cv2.putText(frame, status_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("AI Smart Attendance System (Kiosk Monitor)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('r'), ord('R')):
            known_users = database.get_all_users()
            print(f"Reloaded database: {len(known_users)} users found.")

    cap.release()
    cv2.destroyAllWindows()
    print("Kiosk closed successfully.")

if __name__ == "__main__":
    cam_id = 0
    if len(sys.argv) > 1:
        try:
            cam_id = int(sys.argv[1])
        except ValueError:
            cam_id = sys.argv[1] # Support RTSP stream url string
    run_kiosk(cam_id)
