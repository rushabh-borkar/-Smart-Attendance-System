import sqlite3
import datetime
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from config import DATABASE_PATH, DEFAULT_LATE_CUTOFF_TIME, DEFAULT_SIMILARITY_THRESHOLD

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_code TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        department TEXT DEFAULT 'General',
        role TEXT DEFAULT 'Student',
        email TEXT,
        photo_path TEXT,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create Attendance table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        status TEXT NOT NULL,
        confidence REAL DEFAULT 0.0,
        snapshot_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        UNIQUE(user_id, date)
    )
    """)

    # Create Settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)

    # Insert default settings if not exist
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('late_cutoff', ?)", (DEFAULT_LATE_CUTOFF_TIME,))
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('similarity_threshold', ?)", (str(DEFAULT_SIMILARITY_THRESHOLD),))

    conn.commit()
    conn.close()

def set_setting(key: str, value: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def get_setting(key: str, default: str = "") -> str:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else default

def add_user(
    user_code: str,
    full_name: str,
    department: str,
    role: str,
    email: str,
    photo_path: str,
    embedding: np.ndarray
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    
    # Store embedding as binary bytes
    emb_bytes = embedding.astype(np.float32).tobytes()
    
    cursor.execute("""
        INSERT INTO users (user_code, full_name, department, role, email, photo_path, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_code.strip(), full_name.strip(), department.strip(), role.strip(), email.strip(), photo_path, emb_bytes))
    
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id

def get_all_users() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_code, full_name, department, role, email, photo_path, embedding, created_at FROM users ORDER BY full_name ASC")
    rows = cursor.fetchall()
    users = []
    for r in rows:
        emb_arr = np.frombuffer(r["embedding"], dtype=np.float32)
        users.append({
            "id": r["id"],
            "user_code": r["user_code"],
            "full_name": r["full_name"],
            "department": r["department"],
            "role": r["role"],
            "email": r["email"],
            "photo_path": r["photo_path"],
            "embedding": emb_arr,
            "created_at": r["created_at"]
        })
    conn.close()
    return users

def get_user_by_code(user_code: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_code = ?", (user_code.strip(),))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "user_code": row["user_code"],
        "full_name": row["full_name"],
        "department": row["department"],
        "role": row["role"],
        "email": row["email"],
        "photo_path": row["photo_path"],
        "embedding": np.frombuffer(row["embedding"], dtype=np.float32),
        "created_at": row["created_at"]
    }

def delete_user(user_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def check_attendance_today(user_id: int) -> Optional[Dict[str, Any]]:
    today_str = datetime.date.today().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.*, u.full_name, u.user_code, u.department 
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE a.user_id = ? AND a.date = ?
    """, (user_id, today_str))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def record_attendance(user_id: int, confidence: float = 1.0, snapshot_path: str = "") -> Tuple[bool, str, Dict[str, Any]]:
    now = datetime.datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_time_str = now.strftime("%H:%M:%S")

    # Check if already marked today
    existing = check_attendance_today(user_id)
    if existing:
        return False, f"Attendance already recorded today at {existing['time']}.", existing

    # Check late cutoff
    late_cutoff = get_setting("late_cutoff", DEFAULT_LATE_CUTOFF_TIME)
    status = "Present"
    try:
        cutoff_dt = datetime.datetime.strptime(late_cutoff, "%H:%M:%S").time()
        if now.time() > cutoff_dt:
            status = "Late"
    except Exception:
        status = "Present"

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO attendance (user_id, date, time, status, confidence, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, today_str, current_time_str, status, round(float(confidence), 2), snapshot_path))
        conn.commit()
        
        cursor.execute("""
            SELECT a.*, u.full_name, u.user_code, u.department, u.role
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE a.id = ?
        """, (cursor.lastrowid,))
        record = dict(cursor.fetchone())
        conn.close()
        return True, f"Attendance marked ({status}) successfully!", record
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Attendance already recorded for today.", {}

def get_today_attendance() -> List[Dict[str, Any]]:
    today_str = datetime.date.today().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.id, a.date, a.time, a.status, a.confidence,
               u.id as user_id, u.user_code, u.full_name, u.department, u.role
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE a.date = ?
        ORDER BY a.time DESC
    """, (today_str,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_attendance_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    department: Optional[str] = None,
    user_code: Optional[str] = None
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT a.id, a.date, a.time, a.status, a.confidence,
               u.id as user_id, u.user_code, u.full_name, u.department, u.role
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE 1=1
    """
    params = []
    
    if start_date:
        query += " AND a.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND a.date <= ?"
        params.append(end_date)
    if department and department != "All":
        query += " AND u.department = ?"
        params.append(department)
    if user_code and user_code.strip():
        query += " AND (u.user_code LIKE ? OR u.full_name LIKE ?)"
        pattern = f"%{user_code.strip()}%"
        params.extend([pattern, pattern])
        
    query += " ORDER BY a.date DESC, a.time DESC"
    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_absent_users_today() -> List[Dict[str, Any]]:
    today_str = datetime.date.today().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.user_code, u.full_name, u.department, u.role, u.email
        FROM users u
        WHERE u.id NOT IN (
            SELECT user_id FROM attendance WHERE date = ?
        )
        ORDER BY u.full_name ASC
    """, (today_str,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

# Initialize DB on load
init_db()
