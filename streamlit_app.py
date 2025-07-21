# Full Streamlit app with MediaPipe integration, jump height calculation, and local history storage

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import tempfile
import os
import json
from datetime import datetime

st.set_page_config(page_title="Highland Dance Tracker", layout="centered")
st.title("🏴 Highland Dance Performance Tracker")

# Storage path for session history
SESSION_DIR = "dance_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Helper: Extract jump height from ankle Y displacement
def calculate_jump_heights(landmarks_sequence, frame_height):
    jump_heights = []
    for frame_landmarks in landmarks_sequence:
        try:
            left_ankle_y = frame_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            right_ankle_y = frame_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
            jump_heights.append((1 - avg_ankle_y) * frame_height)
        except:
            jump_heights.append(None)
    return jump_heights

# Upload video
video_file = st.file_uploader("📄 Upload your dance video (.mp4)", type=[".mp4"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    st.video(tfile.name)

    if st.button("Run Analysis"):
        st.info("Extracting pose landmarks and calculating jump heights...")

        cap = cv2.VideoCapture(tfile.name)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        landmarks_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks_sequence.append(results.pose_landmarks.landmark)

        cap.release()

        jump_heights_raw = calculate_jump_heights(landmarks_sequence, frame_height)
        jump_heights_clean = [h for h in jump_heights_raw if h is not None]

        if not jump_heights_clean:
            st.error("No valid jump data found.")
        else:
            # Summary stats
            avg_jump = round(np.mean(jump_heights_clean), 2)
            max_jump = round(np.max(jump_heights_clean), 2)
            st.success("✅ Analysis complete!")

            st.subheader("📈 Jump Height Over Time")
            st.line_chart(jump_heights_clean)

            st.markdown(f"- **Average Jump Height:** {avg_jump} px")
            st.markdown(f"- **Max Jump Height:** {max_jump} px")

            # User rating
            st.subheader("🗳️ Rate This Performance")
            rating = st.radio("How did this session feel?", ["Excellent", "Good", "Average", "Needs Improvement"])
            comments = st.text_area("Any additional notes?")

            if st.button("📅 Save Session"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_data = {
                    "timestamp": timestamp,
                    "avg_jump_height": avg_jump,
                    "max_jump_height": max_jump,
                    "rating": rating,
                    "comments": comments,
                    "jump_heights": jump_heights_clean
                }
                filepath = os.path.join(SESSION_DIR, f"session_{timestamp}.json")
                with open(filepath, "w") as f:
                    json.dump(session_data, f, indent=4)
                st.success(f"Session saved to: {filepath}")

# Load and display session history
st.sidebar.title("📚 Session History")
session_files = [f for f in os.listdir(SESSION_DIR) if f.endswith(".json")]

if session_files:
    selected_session = st.sidebar.selectbox("Select a session", session_files)
    with open(os.path.join(SESSION_DIR, selected_session)) as f:
        session_data = json.load(f)
        st.sidebar.markdown(f"**Date:** {session_data['timestamp']}")
        st.sidebar.markdown(f"**Avg Jump:** {session_data['avg_jump_height']} px")
        st.sidebar.markdown(f"**Max Jump:** {session_data['max_jump_height']} px")
        st.sidebar.markdown(f"**Rating:** {session_data['rating']}")
        st.sidebar.markdown(f"**Comments:** {session_data['comments']}")
else:
    st.sidebar.info("No saved sessions yet.")