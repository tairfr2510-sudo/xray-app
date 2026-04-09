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

# הגדרות דף
st.set_page_config(page_title="Smart X-Ray System", layout="wide")

st.markdown("""
    <style>
    body { background-color: #0a0e27; color: #00BFFF; }
    .main { background-color: #0a0e27; }
    h1 { color: #00BFFF; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🩻 מערכת רנטגן חכמה")

# --- לוגיקה רפואית (נשארת אותו דבר) ---
TARGET_MAPPING = {
    "chest": [11, 12], "abdomen": [23, 24], "head": [0, 7, 8],
    "right_shoulder": [12], "right_elbow": [14], "right_wrist": [16],
    "right_hip": [24], "right_knee": [26], "right_ankle": [28],
    "left_shoulder": [11], "left_elbow": [13], "left_wrist": [15],
    "left_hip": [23], "left_knee": [25], "left_ankle": [27]
}

def hebrew_to_target(text):
    text = text.lower()
    if "ברך" in text: return "right_knee" if "ימין" in text else "left_knee"
    if "חזה" in text: return "chest"
    if "בטן" in text or "אגן" in text: return "abdomen"
    if "ראש" in text: return "head"
    return "chest"

def calculate_limb_angle(pose_landmarks, target_name, width, height):
    angle_pairs = {"right_knee": (24, 26), "left_knee": (23, 25), "right_wrist": (14, 16), "left_wrist": (13, 15)}
    if target_name not in angle_pairs: return 0.0
    idx1, idx2 = angle_pairs[target_name]; p1, p2 = pose_landmarks[idx1], pose_landmarks[idx2]
    angle_rad = math.atan2((p2.y * height) - (p1.y * height), (p2.x * width) - (p1.x * width))
    return round(math.degrees(angle_rad) - 90, 1)

# --- הגדרת מודל MediaPipe ---
model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)

# --- Sidebar ---
st.sidebar.header("⚙️ הגדרות")
medical_problem = st.sidebar.text_area("בעיה רפואית", placeholder="למשל: ברך ימין")
show_skeleton = st.sidebar.checkbox("🦴 הצג שלד", value=True)

# --- פונקציית עיבוד הווידאו (WebRTC) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    height, width, _ = img.shape
    
    # אתחול המודל בתוך ה-Callback (חשוב ל-WebRTC)
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = landmarker.detect(mp_image)
        
        target_name = hebrew_to_target(medical_problem) if medical_problem else None
        
        if target_name and results.pose_landmarks:
            indices = TARGET_MAPPING.get(target_name, [0])
            landmark_list = results.pose_landmarks[0]
            t_x = int((sum(landmark_list[i].x for i in indices) / len(indices)) * width)
            t_y = int((sum(landmark_list[i].y for i in indices) / len(indices)) * height)
            
            # ציור כוונת
            cv2.circle(img, (t_x, t_y), 25, (0, 255, 0), 2)
            cv2.line(img, (t_x-30, t_y), (t_x+30, t_y), (0, 0, 255), 2)
            cv2.line(img, (t_x, t_y-30), (t_x, t_y+30), (0, 0, 255), 2)
            
        if show_skeleton and results.pose_landmarks:
            for lm in results.pose_landmarks[0]:
                if lm.visibility > 0.5:
                    cv2.circle(img, (int(lm.x*width), int(lm.y*height)), 3, (255, 0, 255), -1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- תצוגה ראשית ---
st.subheader("📹 מצלמת רנטגן חיה")
webrtc_streamer(
    key="xray-filter",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)