import os
import sys
import urllib.request
import cv2
import numpy as np
import datetime
import random
from pathlib import Path

# Configure utf-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
import database
import face_engine

# High quality royalty-free student portraits from public CDN
STUDENT_DATASET = [
    {
        "user_code": "21CS101",
        "name": "Aarav Sharma",
        "department": "Computer Science",
        "role": "Student",
        "email": "aarav.sharma@college.edu",
        "url": "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400&auto=format&fit=crop&q=80"
    },
    {
        "user_code": "21CS102",
        "name": "Priya Patel",
        "department": "Computer Science",
        "role": "Student",
        "email": "priya.patel@college.edu",
        "url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&auto=format&fit=crop&q=80"
    },
    {
        "user_code": "21CS103",
        "name": "Rohan Verma",
        "department": "Computer Science",
        "role": "Student",
        "email": "rohan.verma@college.edu",
        "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&auto=format&fit=crop&q=80"
    },
    {
        "user_code": "21IT201",
        "name": "Ananya Iyer",
        "department": "Information Technology",
        "role": "Student",
        "email": "ananya.iyer@college.edu",
        "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&auto=format&fit=crop&q=80"
    },
    {
        "user_code": "21IT202",
        "name": "Siddharth Rao",
        "department": "Information Technology",
        "role": "Student",
        "email": "siddharth.rao@college.edu",
        "url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&auto=format&fit=crop&q=80"
    },
    {
        "user_code": "21AI301",
        "name": "Sneha Kulkarni",
        "department": "AI & Data Science",
        "role": "Student",
        "email": "sneha.kulkarni@college.edu",
        "url": "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&auto=format&fit=crop&q=80"
    },
    {
        "user_code": "21AI302",
        "name": "Arjun Nair",
        "department": "AI & Data Science",
        "role": "Student",
        "email": "arjun.nair@college.edu",
        "url": "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=400&auto=format&fit=crop&q=80"
    },
    {
        "user_code": "21EC401",
        "name": "Neha Gupta",
        "department": "Electronics & Comm.",
        "role": "Student",
        "email": "neha.gupta@college.edu",
        "url": "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400&auto=format&fit=crop&q=80"
    },
    {
        "user_code": "21EC402",
        "name": "Vikram Joshi",
        "department": "Electronics & Comm.",
        "role": "Student",
        "email": "vikram.joshi@college.edu",
        "url": "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400&auto=format&fit=crop&q=80"
    },
    {
        "user_code": "FAC001",
        "name": "Dr. Rajesh Kumar",
        "department": "Computer Science",
        "role": "Faculty",
        "email": "rajesh.kumar@college.edu",
        "url": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&auto=format&fit=crop&q=80"
    }
]

def seed_demo_data():
    print("=" * 60)
    print("🎓 SEEDING COLLEGE DEMO DATASET & HISTORICAL ATTENDANCE")
    print("=" * 60)

    database.init_db()
    engine = face_engine.FaceEngine()
    
    headers = {"User-Agent": "Mozilla/5.0"}
    created_user_ids = []

    for item in STUDENT_DATASET:
        code = item["user_code"]
        existing = database.get_user_by_code(code)
        if existing:
            print(f"⏩ User {code} ({item['name']}) already enrolled. Skipping enrollment.")
            created_user_ids.append((existing["id"], item))
            continue

        print(f"📥 Downloading portrait for {code}: {item['name']}...")
        photo_filename = f"{code}_{item['name'].replace(' ', '_')}.jpg"
        photo_path = config.FACES_DIR / photo_filename

        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                img_bytes = resp.read()
            
            with open(photo_path, "wb") as f:
                f.write(img_bytes)

            # Read image and extract face embedding
            img = cv2.imread(str(photo_path))
            if img is None:
                print(f"❌ Failed to decode image for {code}")
                continue

            emb, face, msg = engine.get_primary_face_embedding(img)
            if emb is None:
                # If face angle was off, try detecting any face in the frame
                faces = engine.detect_faces(img)
                if faces:
                    emb = engine.extract_embedding(img, faces[0])
                else:
                    print(f"⚠️ Face detection failed on image for {code}: {msg}")
                    continue

            user_id = database.add_user(
                user_code=code,
                full_name=item["name"],
                department=item["department"],
                role=item["role"],
                email=item["email"],
                photo_path=str(photo_path),
                embedding=emb
            )
            print(f"✅ Enrolled {code} - {item['name']} [ID: #{user_id}]")
            created_user_ids.append((user_id, item))

        except Exception as e:
            print(f"❌ Error enrolling {code}: {e}")

    # Generate 14 days of realistic past attendance
    print("\n📅 Generating past 14 days attendance records...")
    today = datetime.date.today()
    conn = database.get_connection()
    c = conn.cursor()

    for days_ago in range(14, -1, -1):
        target_date = today - datetime.timedelta(days=days_ago)
        
        # Skip Sundays (college holiday)
        if target_date.weekday() == 6:
            continue

        date_str = target_date.isoformat()

        for user_id, u_info in created_user_ids:
            # 88% attendance rate (some absentees randomly)
            if random.random() < 0.12 and days_ago > 0:
                continue # Student was absent on this day

            # Random arrival time between 08:45 AM and 09:55 AM
            hour = 8 if random.random() < 0.3 else 9
            minute = random.randint(45, 59) if hour == 8 else random.randint(0, 50)
            second = random.randint(0, 59)
            time_str = f"{hour:02d}:{minute:02d}:{second:02d}"

            # If arrival is after 09:30 -> Late
            status = "Late" if (hour == 9 and minute > 30) else "Present"
            confidence = round(random.uniform(0.78, 0.96), 2)

            try:
                c.execute("""
                    INSERT OR IGNORE INTO attendance (user_id, date, time, status, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, date_str, time_str, status, confidence))
            except Exception:
                pass

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("🎉 DEMO COLLEGE DATASET CREATED SUCCESSFULLY!")
    print(f"👥 Enrolled Profiles: {len(created_user_ids)} students & faculty")
    print("📊 Generated 14 days of realistic attendance history")
    print("=" * 60)
    print("\n👉 IMPORTANT STEP FOR YOUR LIVE DEMO:")
    print("1. Launch the web app:  python -m streamlit run app.py")
    print("2. Go to '👤 Register New Person'")
    print("3. Enter your own Name & Roll Number and take a picture with your webcam!")
    print("4. Now when you scan in '📸 Live Attendance', your own face will be recognized live in front of the panel!")
    print("=" * 60)

if __name__ == "__main__":
    seed_demo_data()
