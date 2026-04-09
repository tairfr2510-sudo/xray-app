import streamlit as st
import cv2 
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import math
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ==================== הגדרות עמוד ועיצוב ====================
st.set_page_config(page_title="Smart X-Ray System", layout="wide")

st.markdown("""
    <style>
    body { background-color: #0a0e27; color: #00BFFF; }
    .main { background-color: #0a0e27; }
    h1 { color: #00BFFF; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🩻 מערכת רנטגן חכמה (WebRTC)")

# ==================== קבועים מהקוד המקורי ====================
STATE_IDLE = 0
STATE_MACRO_MOVE = 1
STATE_MICRO_CENTER = 2
STATE_READY = 3

STATE_NAMES = {
    STATE_IDLE: "IDLE",
    STATE_MACRO_MOVE: "MACRO MOVE",
    STATE_MICRO_CENTER: "MICRO CENTER",
    STATE_READY: "READY"
}

CENTER_TOLERANCE = 20  

TARGET_MAPPING = {
    "chest": [11, 12], "abdomen": [23, 24], "head": [0, 7, 8],
    "right_shoulder": [12], "right_elbow": [14], "right_wrist": [16],
    "right_hip": [24], "right_knee": [26], "right_ankle": [28],
    "left_shoulder": [11], "left_elbow": [13], "left_wrist": [15],
    "left_hip": [23], "left_knee": [25], "left_ankle": [27]
}

# ==================== פונקציות לוגיקה מקוריות ====================
def hebrew_to_target(text):
    text = text.lower()
    if "knee" in text: return "right_knee" if "right" in text else "left_knee"
    if "ankle" in text: return "right_ankle" if "right" in text else "left_ankle"
    if "shoulder" in text: return "right_shoulder" if "right" in text else "left_shoulder"
    if "elbow" in text: return "right_elbow" if "right" in text else "left_elbow"
    if "wrist" in text: return "right_wrist" if "right" in text else "left_wrist"
    if "hip" in text: return "right_hip" if "right" in text else "left_hip"
    if "chest" in text: return "chest"
    if "head" in text: return "head"
    if "abdo" in text or "belly" in text or "stomach" in text: return "abdomen"
    
    if "ברך" in text: return "right_knee" if "ימין" in text else "left_knee"
    if "חזה" in text: return "chest"
    if "בטן" in text or "אגן" in text: return "abdomen"
    if "ראש" in text: return "head"
    if "קרסול" in text: return "right_ankle" if "ימין" in text else "left_ankle"
    if "כתף" in text: return "right_shoulder" if "ימין" in text else "left_shoulder"
    if "מרפק" in text: return "right_elbow" if "ימין" in text else "left_elbow"
    if "פרק יד" in text or "יד" in text: return "right_wrist" if "ימין" in text else "left_wrist"
    if "כסל" in text or "ירך" in text: return "right_hip" if "ימין" in text else "left_hip"
    return "chest"

def extract_medical_protocol(text):
    text = text.lower()
    protocols = {
        "fracture": {"keywords": ["שבר", "סדק", "ריסוק", "fracture"], "view": "AP + Lateral", "tilt": "0 deg AND 90 deg"},
        "sprain": {"keywords": ["נקע", "רצועות", "עיקום", "sprain"], "view": "AP + Mortise/Oblique", "tilt": "0 deg AND 15-20 deg"},
        "fluid": {"keywords": ["נוזל", "נפיחות", "כדורית", "swelling"], "view": "Weight-bearing AP + Lateral", "tilt": "0 deg AND 90 deg"},
        "foreign_body": {"keywords": ["חפץ", "זכוכית", "מתכת", "חד"], "view": "AP + Tangential", "tilt": "0 deg AND Multi-angle"}
    }
    for condition, data in protocols.items():
        if any(keyword in text for keyword in data["keywords"]):
            return data["view"], data["tilt"], condition
    return "Standard AP", "0 deg", "default_pain"

def calculate_limb_angle(pose_landmarks, target_name, width, height):
    angle_pairs = {
        "right_ankle": (26, 28), "left_ankle": (25, 27),
        "right_knee": (24, 26), "left_knee": (23, 25),
        "right_wrist": (14, 16), "left_wrist": (13, 15),
        "right_elbow": (12, 14), "left_elbow": (11, 13)
    }
    if target_name not in angle_pairs: return 0.0 
    idx1, idx2 = angle_pairs[target_name]
    p1, p2 = pose_landmarks[idx1], pose_landmarks[idx2]
    dy = (p2.y * height) - (p1.y * height)
    dx = (p2.x * width) - (p1.x * width)
    angle_rad = math.atan2(dy, dx)
    return round(math.degrees(angle_rad) - 90, 1)

def apply_dynamic_zoom(image, zoom_level, zoom_center_x, zoom_center_y):
    if zoom_level <= 1.0: return image
    height, width = image.shape[:2]
    crop_width = int(width / zoom_level)
    crop_height = int(height / zoom_level)
    x1 = max(0, min(zoom_center_x - crop_width // 2, width - crop_width))
    y1 = max(0, min(zoom_center_y - crop_height // 2, height - crop_height))
    x2 = min(width, x1 + crop_width)
    y2 = min(height, y1 + crop_height)
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (width, height))

def control_loop_logic(target_coords, frame_width, frame_height, current_state):
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    error_x = target_coords[0] - frame_center_x
    error_y = target_coords[1] - frame_center_y
    error_distance = (error_x**2 + error_y**2) ** 0.5
    
    if current_state == STATE_IDLE:
        return STATE_IDLE, "IDLE - Waiting for target"
    elif current_state == STATE_MACRO_MOVE:
        if error_distance < 100:
            return STATE_MICRO_CENTER, f"MICRO_CENTER initiated - Error: {error_distance:.1f}px"
        else:
            dir_str = ("LEFT " if error_x < -30 else "RIGHT " if error_x > 30 else "") + \
                      ("UP" if error_y < -30 else "DOWN" if error_y > 30 else "")
            return STATE_MACRO_MOVE, f"MACRO_MOVE: {dir_str.strip()} | Error: {error_distance:.1f}px"
    elif current_state == STATE_MICRO_CENTER:
        if error_distance <= CENTER_TOLERANCE:
            return STATE_READY, f"READY - Target locked! (+-{CENTER_TOLERANCE}px)"
        else:
            return STATE_MICRO_CENTER, f"MICRO_CENTER adjusting | Error: {error_distance:.1f}px"
    elif current_state == STATE_READY:
        if error_distance > CENTER_TOLERANCE + 10:
            return STATE_MACRO_MOVE, "Lost lock - returning to MACRO_MOVE"
        return STATE_READY, f"READY - Holding | Error: {error_distance:.1f}px"
    return STATE_IDLE, "ERROR"

# ==================== התקנת המודל ====================
model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)

# ==================== בניית הממשק השמאלי (Sidebar) ====================
st.sidebar.header("⚙️ בקרה וקלט")
medical_problem = st.sidebar.text_area("הזן בעיה רפואית", placeholder="לדוגמה: חשד לנקע בברך ימין")
show_skeleton = st.sidebar.checkbox("🦴 הצג שלד", value=False)
is_recording = st.sidebar.checkbox("⏺ מצב הקלטה / זום אוטומטי", value=False)

# ==================== מעבד הווידאו המותאם ל-WebRTC ====================
class XRayVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.zoom_level = 1.0
        self.current_state = STATE_IDLE
        self.ready_frame_count = 0
        
        # אתחול המודל 
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        height, width, _ = img.shape
        
        # קבלת הקלט מהמשתמש (הועבר דרך הממשק)
        target_name = hebrew_to_target(medical_problem) if medical_problem else None
        
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)
        
        motor_command = ""
        tube_command = ""
        
        if target_name and results.pose_landmarks:
            indices = TARGET_MAPPING.get(target_name, [0])
            sum_x = sum(results.pose_landmarks[0][i].x for i in indices)
            sum_y = sum(results.pose_landmarks[0][i].y for i in indices)
            
            target_x = int((sum_x / len(indices)) * width)
            target_y = int((sum_y / len(indices)) * height)
            
            if is_recording:
                if self.zoom_level < 4.0:
                    self.zoom_level += 0.05
                img = apply_dynamic_zoom(img, self.zoom_level, target_x, target_y)
                
                if self.current_state == STATE_IDLE:
                    self.current_state = STATE_MACRO_MOVE
                    
                self.current_state, motor_command = control_loop_logic(
                    (width//2, height//2), width, height, self.current_state
                )
                
                if self.current_state == STATE_READY:
                    rot_angle = calculate_limb_angle(results.pose_landmarks[0], target_name, width, height)
                    view_name, tube_tilt, _ = extract_medical_protocol(medical_problem)
                    motor_command = f"Rot: {rot_angle}deg | Mode: {view_name}"
                    tube_command = f"Tilt: {tube_tilt}"
            else:
                self.zoom_level = 1.0
                self.current_state = STATE_IDLE
                motor_command = "Check 'Recording' to start Auto-Zoom"
                
        else:
            self.zoom_level = 1.0
            self.current_state = STATE_IDLE

        # --- שכבות תצוגה ---
        # 1. טקסטים
        if target_name:
            cv2.putText(img, f"Target: {target_name.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(img, f"Zoom: {self.zoom_level:.1f}x", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if motor_command:
                cv2.putText(img, motor_command, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if tube_command:
                cv2.putText(img, tube_command, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
        # 2. שלד
        if show_skeleton and results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            
            # מערך החיבורים המקורי שלך
            CONNECTIONS = [
                (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10),
                (11,12), (23,24), (11,23), (12,24), (11,13), (13,15), (15,17), 
                (15,19), (15,21), (17,19), (12,14), (14,16), (16,18), (16,20), 
                (16,22), (18,20), (23,25), (25,27), (27,29), (29,31), (31,27),
                (24,26), (26,28), (28,30), (30,32), (32,28)
            ]
            
            # ציור הקווים המחברים (טורקיז)
            for start, end in CONNECTIONS:
                if start < len(landmarks) and end < len(landmarks):
                    lm_start, lm_end = landmarks[start], landmarks[end]
                    if lm_start.visibility > 0.3 and lm_end.visibility > 0.3:
                        start_pos = (int(lm_start.x * width), int(lm_start.y * height))
                        end_pos = (int(lm_end.x * width), int(lm_end.y * height))
                        cv2.line(img, start_pos, end_pos, (0, 255, 255), 2)
            
            # ציור המפרקים בצבעים הנכונים
            for idx, lm in enumerate(landmarks):
                if lm.visibility > 0.3:
                    x, y = int(lm.x * width), int(lm.y * height)
                    if idx <= 10: color = (255, 255, 0)      # ראש
                    elif idx in [11, 12]: color = (0, 255, 0) # כתפיים
                    elif 13 <= idx <= 22: color = (255, 0, 255) # זרועות
                    else: color = (255, 0, 0)                 # רגליים ואגן
                    cv2.circle(img, (x, y), 3, color, -1)

                    
        # 3. כוונת אמצע תמיד במרכז
        cx, cy = width // 2, height // 2
        cv2.line(img, (cx - 40, cy), (cx + 40, cy), (0, 0, 255), 3)
        cv2.line(img, (cx, cy - 40), (cx, cy + 40), (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), 25, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================== תצוגה ראשית ====================
col1, col2 = st.columns([2, 1])
with col1:
    webrtc_streamer(
        key="xray-system",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=XRayVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280, "min": 640},
                "height": {"ideal": 720, "min": 480}
            },
            "audio": False
        },
        async_processing=True,
    )
with col2:
    st.info("👈 **הוראות:**\n1. אשר שימוש במצלמה.\n2. הקלד בעיה רפואית בתפריט הצד.\n3. סמן את תיבת ה'הקלטה' כדי להפעיל את מכונת המצבים והזום האוטומטי.")