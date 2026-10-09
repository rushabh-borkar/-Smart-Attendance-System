import os
import sys
import cv2
from pathlib import Path
import config
import database
import face_engine

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def batch_import_from_folder(folder_path="batch_students"):
    """
    Imports images from a folder where each filename is formatted as:
    RollNo_FullName_Department.jpg
    Example:
    21CS205_Rahul_Sharma_Computer_Science.jpg
    or simply:
    21CS205_Rahul_Sharma.jpg
    """
    p = Path(folder_path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        print(f"Created folder '{folder_path}'. Place student images inside this folder and run again.")
        print("Filename naming convention: RollNumber_FullName_Department.jpg")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [f for f in p.iterdir() if f.suffix.lower() in valid_exts]

    if not image_files:
        print(f"No image files found in '{folder_path}'.")
        print("Drop images with format: RollNo_FullName_Department.jpg")
        return

    database.init_db()
    engine = face_engine.FaceEngine()

    print(f"Found {len(image_files)} images to process...\n")
    success_count = 0

    for img_path in image_files:
        filename_stem = img_path.stem
        parts = filename_stem.split("_")

        if len(parts) >= 2:
            user_code = parts[0].strip()
            # If department was included as last part
            if len(parts) >= 3:
                department = parts[-1].replace("-", " ")
                full_name = " ".join(parts[1:-1]).replace("-", " ")
            else:
                department = "General"
                full_name = " ".join(parts[1:]).replace("-", " ")
        else:
            user_code = f"STU-{abs(hash(filename_stem)) % 10000:04d}"
            full_name = filename_stem.replace("-", " ").title()
            department = "General"

        existing = database.get_user_by_code(user_code)
        if existing:
            print(f"⏩ User {user_code} ({existing['full_name']}) already registered. Skipping.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Could not read image: {img_path.name}")
            continue

        emb, face, msg = engine.get_primary_face_embedding(img)
        if emb is None:
            faces = engine.detect_faces(img)
            if faces:
                emb = engine.extract_embedding(img, faces[0])
            else:
                print(f"⚠️ No face detected in: {img_path.name} ({msg})")
                continue

        dest_photo_path = config.FACES_DIR / img_path.name
        cv2.imwrite(str(dest_photo_path), img)

        user_id = database.add_user(
            user_code=user_code,
            full_name=full_name,
            department=department,
            role="Student",
            email=f"{user_code.lower()}@college.edu",
            photo_path=str(dest_photo_path),
            embedding=emb
        )
        print(f"✅ Enrolled: {user_code} | {full_name} | {department} [ID: #{user_id}]")
        success_count += 1

    print(f"\n🎉 Successfully batch enrolled {success_count} student(s)!")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "batch_students"
    batch_import_from_folder(folder)
