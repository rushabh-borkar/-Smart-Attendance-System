import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import io
import time
from pathlib import Path
from PIL import Image

import config
import database
import face_engine

# --- Page Config ---
st.set_page_config(
    page_title="AI Smart Attendance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #78909C;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E222A;
        border: 1px solid #2E3440;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ECEFF4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #88C0D0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Cache FaceEngine so it doesn't reload models on every interaction
@st.cache_resource
def get_engine():
    return face_engine.FaceEngine()

engine = get_engine()

# --- Sidebar ---
st.sidebar.title("🛡️ Attendance Portal")
st.sidebar.markdown(f"**Date:** {datetime.date.today().strftime('%A, %d %B %Y')}")

menu = st.sidebar.radio(
    "Navigation",
    [
        "📸 Live Attendance",
        "👤 Register New Person",
        "📊 Attendance Logs & Analytics",
        "👥 Enrolled Directory",
        "⚙️ System Settings"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Ensure adequate lighting and face camera directly for optimal recognition.")

# ==============================================================================
# 1. LIVE ATTENDANCE MONITOR
# ==============================================================================
if menu == "📸 Live Attendance":
    st.markdown('<div class="main-header">📸 Smart Attendance Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time facial detection, recognition, and automated logging</div>', unsafe_allow_html=True)

    # Top summary metrics
    today_records = database.get_today_attendance()
    all_users = database.get_all_users()
    total_registered = len(all_users)
    present_count = len(today_records)
    late_count = sum(1 for r in today_records if r["status"] == "Late")
    on_time_count = present_count - late_count
    absent_count = max(0, total_registered - present_count)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Enrolled Individuals</div>
            <div class="metric-value">{total_registered}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Present Today</div>
            <div class="metric-value" style="color: #A3BE8C;">{present_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Late Arrivals</div>
            <div class="metric-value" style="color: #EBCB8B;">{late_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Absentees</div>
            <div class="metric-value" style="color: #BF616A;">{absent_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("📷 Camera Scan")
        st.caption("Point your face towards the camera and capture snapshot to check in.")
        
        sim_threshold = float(database.get_setting("similarity_threshold", str(config.DEFAULT_SIMILARITY_THRESHOLD)))

        camera_photo = st.camera_input("Attendance Scanner", label_visibility="collapsed")
        
        if camera_photo is not None:
            # Convert uploaded bytes to OpenCV BGR image
            bytes_data = camera_photo.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if cv_img is not None:
                faces = engine.detect_faces(cv_img)
                display_img = cv_img.copy()

                if len(faces) == 0:
                    st.warning("⚠️ No face detected in the frame. Please look directly at the camera and try again.")
                else:
                    processed_match = False
                    for face in faces:
                        emb = engine.extract_embedding(cv_img, face)
                        matched_user, score = engine.match_against_known(emb, all_users, threshold=sim_threshold)

                        if matched_user:
                            user_id = matched_user["id"]
                            user_name = matched_user["full_name"]
                            user_code = matched_user["user_code"]
                            confidence_pct = min(100.0, (score / 0.70) * 100.0)

                            # Record attendance
                            success, msg, rec = database.record_attendance(user_id, confidence=score)

                            if success:
                                # First time today
                                status_color = (0, 255, 0) # Green
                                subtitle = f"STATUS: {rec.get('status', 'Present').upper()}"
                                st.success(f"🎉 **Welcome, {user_name}!** {msg} (Confidence: {confidence_pct:.1f}%)")
                            else:
                                # Already marked today
                                status_color = (255, 200, 0) # Cyan/Blue
                                subtitle = "ALREADY RECORDED"
                                st.info(f"ℹ️ **{user_name}**: {msg}")

                            display_img = face_engine.FaceEngine.draw_face_hud(
                                display_img, face, user_name, subtitle, status_color=status_color, confidence_pct=confidence_pct
                            )
                            processed_match = True
                        else:
                            # Unknown person
                            display_img = face_engine.FaceEngine.draw_face_hud(
                                display_img, face, "Unknown Person", "NOT REGISTERED", status_color=(0, 0, 255), confidence_pct=None
                            )

                    if not processed_match:
                        st.error("❌ Face not recognized. Please make sure you are registered in the system.")

                    # Show annotated output preview
                    rgb_preview = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                    st.image(rgb_preview, caption="Detection Analysis & Verification", use_container_width=True)

    with right_col:
        st.subheader("📋 Today's Live Attendance Feed")
        if today_records:
            df_today = pd.DataFrame(today_records)[["time", "user_code", "full_name", "department", "status"]]
            df_today.columns = ["Time", "ID", "Name", "Department", "Status"]
            
            # Format status badges
            st.dataframe(
                df_today,
                use_container_width=True,
                hide_index=True,
                height=380
            )
        else:
            st.info("No attendance recorded yet today. Check-ins will appear here in real-time.")

        st.caption("🚀 Looking for high-speed 60 FPS kiosk mode? You can also run `python kiosk_monitor.py` for continuous webcam monitoring!")

# ==============================================================================
# 2. ENROLLMENT (REGISTER NEW PERSON)
# ==============================================================================
elif menu == "👤 Register New Person":
    st.markdown('<div class="main-header">👤 Register Student / Employee</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enroll new face profile with 128-dimensional AI facial embedding</div>', unsafe_allow_html=True)

    reg_col1, reg_col2 = st.columns([1, 1])

    with reg_col1:
        st.subheader("1. Profile Details")
        user_code = st.text_input("ID / Roll Number / Employee Code *", placeholder="e.g. STU-101 or EMP-202")
        full_name = st.text_input("Full Name *", placeholder="e.g. John Doe")
        
        c_role, c_dept = st.columns(2)
        with c_role:
            role = st.selectbox("Role", ["Student", "Faculty", "Employee", "Visitor", "Other"])
        with c_dept:
            department = st.text_input("Department / Class", value="Computer Science")
            
        email = st.text_input("Email Address (Optional)", placeholder="user@example.com")

    with reg_col2:
        st.subheader("2. Capture or Upload Face Photo")
        photo_mode = st.radio("Photo Source:", ["Webcam Capture", "Upload File"], horizontal=True)

        captured_img = None
        if photo_mode == "Webcam Capture":
            cam_data = st.camera_input("Take Enrollment Photo")
            if cam_data is not None:
                captured_img = cv2.imdecode(np.frombuffer(cam_data.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        else:
            uploaded_file = st.file_uploader("Upload Profile Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                captured_img = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

        if captured_img is not None:
            # Analyze face quality
            emb, face_data, msg = engine.get_primary_face_embedding(captured_img)
            
            preview_img = captured_img.copy()
            if face_data is not None:
                preview_img = face_engine.FaceEngine.draw_face_hud(
                    preview_img, face_data, full_name or "Valid Face", "READY TO ENROLL", status_color=(0, 255, 0)
                )
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

            rgb_preview = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_preview, caption="Enrollment Preview", use_container_width=True)

            # Submit Registration Button
            if st.button("💾 Enroll Profile", type="primary", use_container_width=True):
                if not user_code or not full_name:
                    st.error("Please provide both ID Code and Full Name.")
                elif emb is None:
                    st.error("Cannot enroll without a valid single face detected.")
                else:
                    # Check if ID already exists
                    existing = database.get_user_by_code(user_code)
                    if existing:
                        st.error(f"User with ID '{user_code}' is already registered as '{existing['full_name']}'.")
                    else:
                        # Save photo to disk
                        filename = f"{user_code.strip()}_{int(time.time())}.jpg"
                        photo_file_path = str(config.FACES_DIR / filename)
                        cv2.imwrite(photo_file_path, captured_img)

                        # Save to database
                        user_id = database.add_user(
                            user_code=user_code,
                            full_name=full_name,
                            department=department,
                            role=role,
                            email=email,
                            photo_path=photo_file_path,
                            embedding=emb
                        )
                        st.balloons()
                        st.success(f"🎉 **{full_name}** ({user_code}) has been successfully enrolled! ID: #{user_id}")

# ==============================================================================
# 3. ATTENDANCE LOGS & ANALYTICS
# ==============================================================================
elif menu == "📊 Attendance Logs & Analytics":
    st.markdown('<div class="main-header">📊 Attendance Analytics & Reporting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Audit trails, compliance tracking, and CSV/Excel exports</div>', unsafe_allow_html=True)

    # Filter section
    with st.expander("🔍 Filter Records", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        with f_col1:
            start_date = st.date_input("Start Date", value=datetime.date.today() - datetime.timedelta(days=7))
        with f_col2:
            end_date = st.date_input("End Date", value=datetime.date.today())
        with f_col3:
            all_users = database.get_all_users()
            departments = ["All"] + sorted(list(set(u["department"] for u in all_users if u.get("department"))))
            selected_dept = st.selectbox("Department", departments)
        with f_col4:
            search_query = st.text_input("Search (Name / ID)", placeholder="Enter keyword...")

    # Query attendance records
    records = database.get_attendance_history(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        department=selected_dept,
        user_code=search_query
    )

    if records:
        df = pd.DataFrame(records)
        
        # Metric row
        total_logs = len(df)
        unique_attendees = df["user_id"].nunique()
        late_logs = len(df[df["status"] == "Late"])
        on_time_logs = total_logs - late_logs
        on_time_pct = (on_time_logs / total_logs * 100) if total_logs > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records Found", total_logs)
        m2.metric("Distinct Individuals", unique_attendees)
        m3.metric("On-Time Punctuality", f"{on_time_pct:.1f}%")
        m4.metric("Late Arrivals", late_logs)

        # Tab views: Table vs Visual Charts
        tab_table, tab_charts, tab_absent = st.tabs(["📑 Detailed Logs", "📈 Graphical Analytics", "⚠️ Today's Absentees"])

        with tab_table:
            display_cols = ["date", "time", "user_code", "full_name", "department", "role", "status", "confidence"]
            col_names = ["Date", "Time", "ID", "Name", "Department", "Role", "Status", "Confidence"]
            
            df_display = df[display_cols].copy()
            df_display.columns = col_names

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Export actions
            exp_col1, exp_col2, _ = st.columns([1, 1, 2])
            with exp_col1:
                # CSV Export
                csv_buffer = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_buffer,
                    file_name=f"attendance_report_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with exp_col2:
                # Excel Export
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_display.to_excel(writer, index=False, sheet_name="Attendance")
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_data,
                    file_name=f"attendance_report_{start_date}_{end_date}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        with tab_charts:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.write("**Attendance By Date**")
                daily_counts = df.groupby("date")["id"].count().reset_index()
                daily_counts.columns = ["Date", "Attendees"]
                st.bar_chart(daily_counts.set_index("Date"))

            with chart_col2:
                st.write("**Department Distribution**")
                dept_counts = df.groupby("department")["id"].count().reset_index()
                dept_counts.columns = ["Department", "Check-ins"]
                st.bar_chart(dept_counts.set_index("Department"))

        with tab_absent:
            absentees = database.get_absent_users_today()
            if absentees:
                st.warning(f"🚨 **{len(absentees)}** enrolled individuals have not checked in today.")
                df_abs = pd.DataFrame(absentees)[["user_code", "full_name", "department", "role", "email"]]
                df_abs.columns = ["ID", "Name", "Department", "Role", "Email"]
                st.dataframe(df_abs, use_container_width=True, hide_index=True)
            else:
                st.success("🎉 All enrolled individuals are marked present today!")

    else:
        st.info("No attendance records found matching your filter criteria.")

# ==============================================================================
# 4. ENROLLED DIRECTORY
# ==============================================================================
elif menu == "👥 Enrolled Directory":
    st.markdown('<div class="main-header">👥 Enrolled Profiles Directory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Manage enrolled members, inspect facial profiles, or delete users</div>', unsafe_allow_html=True)

    users = database.get_all_users()
    search_user = st.text_input("🔍 Search User by Name or ID Code:", "")

    if search_user.strip():
        users = [u for u in users if search_user.lower() in u["full_name"].lower() or search_user.lower() in u["user_code"].lower()]

    st.write(f"Showing **{len(users)}** enrolled profiles")

    if users:
        # Display as a grid of cards
        for i in range(0, len(users), 3):
            cols = st.columns(3)
            for j in range(3):
                idx = i + j
                if idx < len(users):
                    u = users[idx]
                    with cols[j]:
                        with st.container(border=True):
                            photo_p = u.get("photo_path")
                            if photo_p and Path(photo_p).exists():
                                st.image(photo_p, width=160)
                            else:
                                st.image("https://via.placeholder.com/160x160.png?text=No+Photo", width=160)

                            st.markdown(f"### {u['full_name']}")
                            st.markdown(f"**ID:** `{u['user_code']}`")
                            st.markdown(f"**Role:** {u['role']} | **Dept:** {u['department']}")
                            if u.get('email'):
                                st.markdown(f"📧 `{u['email']}`")
                            st.caption(f"Enrolled on: {u['created_at']}")

                            if st.button(f"🗑️ Delete User", key=f"del_{u['id']}", type="secondary"):
                                database.delete_user(u["id"])
                                st.warning(f"User {u['full_name']} deleted.")
                                st.rerun()

# ==============================================================================
# 5. SYSTEM SETTINGS
# ==============================================================================
elif menu == "⚙️ System Settings":
    st.markdown('<div class="main-header">⚙️ System Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fine-tune AI detection thresholds and institutional business rules</div>', unsafe_allow_html=True)

    with st.form("settings_form"):
        st.subheader("Business Rules")
        current_late = database.get_setting("late_cutoff", config.DEFAULT_LATE_CUTOFF_TIME)
        late_input = st.text_input("Late Arrival Cutoff Time (HH:MM:SS)", value=current_late)

        st.subheader("AI Recognition Threshold")
        current_thresh = float(database.get_setting("similarity_threshold", str(config.DEFAULT_SIMILARITY_THRESHOLD)))
        thresh_input = st.slider(
            "Cosine Similarity Threshold (Standard: 0.36 - 0.42)",
            min_value=0.25,
            max_value=0.60,
            value=current_thresh,
            step=0.01,
            help="Higher values make recognition stricter (lower false positives, but may reject if lighting is slightly altered)."
        )

        submitted = st.form_submit_button("Save Configuration", type="primary")
        if submitted:
            database.set_setting("late_cutoff", late_input.strip())
            database.set_setting("similarity_threshold", str(thresh_input))
            st.success("✅ Configuration settings updated successfully!")

    st.markdown("---")
    st.subheader("Model Status & System Integrity")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.success(f"**Face Detector:** OpenCV YuNet ONNX (`{Path(config.YUNET_MODEL_PATH).name}`)")
    with m_col2:
        st.success(f"**Face Recognizer:** SFace 128-d ONNX (`{Path(config.SFACE_MODEL_PATH).name}`)")
