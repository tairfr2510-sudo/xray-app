"""
Streamlit Web App - תוכנת רנטגן חכמה בווב
עובדת בטלפון + מחשב + כל מכשיר עם דפדפן
"""

import streamlit as st
import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python
import mediapipe as mp
import math
import os

st.set_page_config(page_title="Smart X-Ray System", layout="wide")

# CSS styling for mobile
st.markdown("""
    <style>
    body { background-color: #0a0e27; color: #00BFFF; }
    .main { background-color: #0a0e27; }
    h1 { color: #00BFFF; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🩻 מערכת רנטגן חכמה")

# Target mapping
TARGET_MAPPING = {
    "chest": [11, 12],
    "abdomen": [23, 24],
    "head": [0, 7, 8],
    "right_shoulder": [12],
    "right_elbow": [14],
    "right_wrist": [16],
    "right_hip": [24],
    "right_knee": [26],
    "right_ankle": [28],
    "left_shoulder": [11],
    "left_elbow": [13],
    "left_wrist": [15],
    "left_hip": [23],
    "left_knee": [25],
    "left_ankle": [27]
}

def hebrew_to_target(text):
    """Convert Hebrew medical terms to target names"""
    text = text.lower()
    
    # Hebrew recognition
    if "ברך" in text:
        if "ימין" in text: return "right_knee"
        if "שמאל" in text: return "left_knee"
        return "right_knee"
    if "חזה" in text: return "chest"
    if "בטן" in text or "אגן" in text: return "abdomen"
    if "ראש" in text: return "head"
    if "קרסול" in text:
        if "ימין" in text: return "right_ankle"
        if "שמאל" in text: return "left_ankle"
        return "right_ankle"
    if "כתף" in text:
        if "ימין" in text: return "right_shoulder"
        if "שמאל" in text: return "left_shoulder"
        return "right_shoulder"
    if "מרפק" in text:
        if "ימין" in text: return "right_elbow"
        if "שמאל" in text: return "left_elbow"
        return "right_elbow"
    if "יד" in text or "פרק יד" in text:
        if "ימין" in text: return "right_wrist"
        if "שמאל" in text: return "left_wrist"
        return "right_wrist"
    
    # English fallback
    if "knee" in text:
        if "right" in text: return "right_knee"
        if "left" in text: return "left_knee"
        return "right_knee"
    if "chest" in text: return "chest"
    if "ankle" in text:
        if "right" in text: return "right_ankle"
        if "left" in text: return "left_ankle"
        return "right_ankle"
    
    return "chest"

def calculate_limb_angle(pose_landmarks, target_name, width, height):
    """Calculate rotation angle for limbs"""
    angle_pairs = {
        "right_ankle": (26, 28),
        "left_ankle": (25, 27),
        "right_knee": (24, 26),
        "left_knee": (23, 25),
        "right_wrist": (14, 16),
        "left_wrist": (13, 15),
        "right_elbow": (12, 14),
        "left_elbow": (11, 13)
    }
    
    if target_name not in angle_pairs:
        return 0.0
    
    idx1, idx2 = angle_pairs[target_name]
    p1 = pose_landmarks[idx1]
    p2 = pose_landmarks[idx2]
    
    dy = (p2.y * height) - (p1.y * height)
    dx = (p2.x * width) - (p1.x * width)
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    normalized_angle = round(angle_deg - 90, 1)
    return normalized_angle

def extract_medical_protocol(text):
    """Extract X-Ray protocol from medical description"""
    text = text.lower()
    
    protocols = {
        "fracture": {"keywords": ["שבר", "סדק", "fracture"], "view": "AP + Lateral", "tilt": "0 deg AND 90 deg"},
        "sprain": {"keywords": ["נקע", "sprain"], "view": "AP + Oblique", "tilt": "0 deg AND 20 deg"},
        "fluid": {"keywords": ["נוזל", "נפיחות", "swelling"], "view": "AP + Lateral", "tilt": "0 deg AND 90 deg"},
    }
    
    for condition, data in protocols.items():
        if any(keyword in text for keyword in data["keywords"]):
            return data["view"], data["tilt"]
    return "Standard AP", "0 deg"

# Initialize MediaPipe
model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    st.info("⬇️ Downloading pose detection model (first time only)...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    st.success("✅ Model downloaded!")

options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5
)

# Sidebar controls
st.sidebar.header("⚙️ הגדרות")
medical_problem = st.sidebar.text_area("בעיה רפואית (בעברית או English)", 
                                       placeholder="דוגמה: ברך ימין, שבר, נקע")

camera_option = st.sidebar.radio("📹 מקור הווידאו", 
                                 ["מצלמת מחשב", "העלה תמונה"])

show_skeleton = st.sidebar.checkbox("🦴 הצג שלד", value=False)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 לייִו ווידאו")
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

with col2:
    st.subheader("📊 נתונים")
    target_display = st.empty()
    angle_display = st.empty()
    protocol_display = st.empty()

# Process video
if camera_option == "מצלמת מחשב":
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ לא ניתן לפתוח את המצלמה")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_count = 0
            max_frames = 30  # Show 30 frames
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ שגיאה בקריאת הווידאו")
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror for selfie mode
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                results = landmarker.detect(mp_image)
                
                # Process results
                target_name = hebrew_to_target(medical_problem) if medical_problem else None
                
                if target_name and results.pose_landmarks and len(results.pose_landmarks) > 0:
                    indices = TARGET_MAPPING[target_name]
                    sum_x = sum(results.pose_landmarks[0][idx].x for idx in indices)
                    sum_y = sum(results.pose_landmarks[0][idx].y for idx in indices)
                    
                    target_x = int((sum_x / len(indices)) * width)
                    target_y = int((sum_y / len(indices)) * height)
                    
                    # Draw crosshair
                    cv2.line(frame, (target_x - 30, target_y), (target_x + 30, target_y), (0, 0, 255), 3)
                    cv2.line(frame, (target_x, target_y - 30), (target_x, target_y + 30), (0, 0, 255), 3)
                    cv2.circle(frame, (target_x, target_y), 20, (0, 255, 0), 2)
                    
                    # Calculate angle
                    angle = calculate_limb_angle(results.pose_landmarks[0], target_name, width, height)
                    view, tilt = extract_medical_protocol(medical_problem)
                    
                    # Display info
                    cv2.putText(frame, f"Target: {target_name.upper()}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Rotation: {angle}°", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    cv2.putText(frame, f"Mode: {view}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
                    # Update sidebar
                    target_display.metric("🎯 מטרה", target_name.replace("_", " ").upper())
                    angle_display.metric("📐 זווית סיבוב", f"{angle}°")
                    protocol_display.metric("🩻 פרוטוקול", view)
                
                # Draw skeleton if enabled
                if show_skeleton and results.pose_landmarks and len(results.pose_landmarks) > 0:
                    for landmark in results.pose_landmarks[0]:
                        if landmark.visibility > 0.3:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
                
                # Display frame
                frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                frame_count += 1
        
        cap.release()

else:  # Upload image
    uploaded_file = st.file_uploader("בחר תמונה", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect(mp_image)
            
            target_name = hebrew_to_target(medical_problem) if medical_problem else None
            
            if target_name and results.pose_landmarks and len(results.pose_landmarks) > 0:
                indices = TARGET_MAPPING[target_name]
                sum_x = sum(results.pose_landmarks[0][idx].x for idx in indices)
                sum_y = sum(results.pose_landmarks[0][idx].y for idx in indices)
                
                target_x = int((sum_x / len(indices)) * width)
                target_y = int((sum_y / len(indices)) * height)
                
                cv2.line(frame, (target_x - 30, target_y), (target_x + 30, target_y), (0, 0, 255), 3)
                cv2.line(frame, (target_x, target_y - 30), (target_x, target_y + 30), (0, 0, 255), 3)
                cv2.circle(frame, (target_x, target_y), 20, (0, 255, 0), 2)
                
                angle = calculate_limb_angle(results.pose_landmarks[0], target_name, width, height)
                view, tilt = extract_medical_protocol(medical_problem)
                
                target_display.metric("🎯 מטרה", target_name.replace("_", " ").upper())
                angle_display.metric("📐 זווית סיבוב", f"{angle}°")
                protocol_display.metric("🩻 פרוטוקול", view)
            
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)

st.info("💡 לעבודה בטלפון: פתח את זה ב-Streamlit Cloud או בשרת משלך")
