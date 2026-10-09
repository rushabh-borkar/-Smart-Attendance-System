import os
import sys
import re
import urllib.request
import cv2
from pathlib import Path
from bs4 import BeautifulSoup

import config
import database
import face_engine

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def fetch_and_enroll_iare_professors():
    print("=" * 65)
    print("🏛️ FETCHING REAL IARE PROFESSORS (DEPT: CSE - AI & ML)")
    print("=" * 65)

    url = "https://www.iare.ac.in/?q=departmentlist/113"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    print(f"Connecting to {url}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"❌ Network fetch failed: {e}")
        # Fallback to cached content
        cached_path = r"C:\Users\Rushabh\.gemini\antigravity\brain\2abdd8ab-1c9f-47aa-ae99-a6571181592f\.system_generated\steps\87\content.md"
        if os.path.exists(cached_path):
            print("Using cached content...")
            with open(cached_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
        else:
            return

    soup = BeautifulSoup(html, "html.parser")
    database.init_db()
    engine = face_engine.FaceEngine()

    faculty_entries = []

    for img in soup.find_all("img"):
        src = img.get("src", "")
        if "faculty_profile_images" in src:
            # Extract employee code from image URL if available, e.g. IARE10960
            emp_match = re.search(r"(IARE\d+)", src, re.IGNORECASE)
            emp_code = emp_match.group(1).upper() if emp_match else ""

            # Extract name and title from container
            container = None
            for p in img.parents:
                classes = str(p.get("class", []))
                if "facInfo" in classes or "views-row" in classes:
                    container = p
                    break

            if container:
                lines = [l.strip() for l in container.get_text(separator="\n", strip=True).splitlines() if l.strip()]
            else:
                lines = []

            # First non-empty line is usually the Name
            full_name = lines[0] if lines else f"Professor {emp_code}"
            # Clean up name if it has trailing characters
            full_name = re.sub(r"\s+", " ", full_name).strip()

            designation = "Professor"
            for line_text in lines[1:5]:
                if any(kw in line_text.lower() for kw in ["professor", "head", "dean"]):
                    designation = line_text
                    break

            faculty_entries.append({
                "code": emp_code or f"IARE_{len(faculty_entries)+1:03d}",
                "name": full_name,
                "designation": designation,
                "img_url": src
            })

    print(f"Found {len(faculty_entries)} professors on IARE portal.\n")
    enrolled_count = 0

    for fac in faculty_entries:
        code = fac["code"]
        name = fac["name"]
        img_url = fac["img_url"]

        # Check if already registered
        existing = database.get_user_by_code(code)
        if existing:
            print(f"⏩ {code} ({name}) already enrolled.")
            continue

        print(f"📥 Processing {code} - {name} ({fac['designation']})...")
        photo_filename = f"{code}_{re.sub(r'[^a-zA-Z0-9]', '_', name)}.jpg"
        photo_path = config.FACES_DIR / photo_filename

        try:
            req = urllib.request.Request(img_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                img_data = resp.read()

            with open(photo_path, "wb") as f:
                f.write(img_data)

            # Read image
            img = cv2.imread(str(photo_path))
            if img is None:
                print(f"   ❌ Could not decode image file")
                continue

            # Detect face & compute 128-d embedding
            faces = engine.detect_faces(img)
            if not faces:
                print(f"   ⚠️ Face detector found no clear frontal face in website thumbnail")
                continue

            # Use largest detected face
            best_face = max(faces, key=lambda f: f[2] * f[3])
            embedding = engine.extract_embedding(img, best_face)

            user_id = database.add_user(
                user_code=code,
                full_name=name,
                department="CSE (AI & ML)",
                role="Faculty",
                email=f"{code.lower()}@iare.ac.in",
                photo_path=str(photo_path),
                embedding=embedding
            )
            print(f"   ✅ Enrolled into Database: [ID: #{user_id}]")
            enrolled_count += 1

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 65)
    print(f"🎉 COMPLETED! Enrolled {enrolled_count} official IARE professors!")
    print("=" * 65)

if __name__ == "__main__":
    fetch_and_enroll_iare_professors()
