import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2
import numpy as np
import pyttsx3
import threading

# -----------------------------------------------
# MODEL LOADING
# Allow user to set model path via environment
# variable or fall back to a local default.
# This makes the app portable for anyone who
# clones the repo — they just set their own path.
# -----------------------------------------------
# Go one folder up from App.py to find best.pt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "best.pt"))

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at: {MODEL_PATH}. "
             f"Please set the MODEL_PATH environment variable or "
             f"place best.pt in the same folder as App.py.")
    st.stop()

model = YOLO(MODEL_PATH)

# Confidence threshold — detections below this
# value will be ignored to avoid false alerts
CONFIDENCE_THRESHOLD = 0.5

# Only run YOLO every Nth frame to keep
# processing fast and avoid timeout
FRAME_SKIP = 10

# -----------------------------------------------
# PAGE CONFIG
# -----------------------------------------------
st.set_page_config(
    page_title="Driver Behaviour Analyser",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Driver Behaviour Analyser")
st.write("Upload a dashcam video to analyse driver behaviour and hear real time voice alerts")


# -----------------------------------------------
# HELPER: Speak text using pyttsx3
# Runs in a background thread so the app
# does not freeze while the voice is speaking
# -----------------------------------------------
def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)   # words per minute
        engine.setProperty('volume', 1.0)  # max volume
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()


# -----------------------------------------------
# HELPER: Map detected sign label to a
# human friendly voice alert message.
# Returns None if the sign is not recognised.
# -----------------------------------------------
def get_alert_message(sign):
    sign = sign.lower()
    if "stop" in sign:
        return "Stop sign detected. Please come to a full stop."
    elif "red" in sign:
        return "Red light detected. Please stop immediately."
    elif "green" in sign:
        return "Green light. You may proceed."
    elif "20" in sign:
        return "Speed limit is 20. Please slow down."
    elif "30" in sign:
        return "Speed limit is 30. Please slow down."
    elif "40" in sign:
        return "Speed limit is 40. Please reduce your speed."
    elif "50" in sign:
        return "Speed limit is 50. Adjust your speed."
    elif "60" in sign:
        return "Speed limit is 60."
    elif "70" in sign:
        return "Speed limit is 70."
    elif "80" in sign:
        return "Speed limit is 80."
    elif "90" in sign:
        return "Speed limit is 90."
    elif "100" in sign:
        return "Speed limit is 100."
    else:
        return None


# -----------------------------------------------
# HELPER: Estimate vehicle motion between two
# consecutive frames using Farneback Optical Flow.
# Returns a magnitude value:
#   Low  (~0-1) = vehicle likely stopped
#   High (~3+)  = vehicle moving fast
# -----------------------------------------------
def estimate_motion(prev_frame, curr_frame):
    try:
        # Convert frames to grayscale for optical flow
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow between frames
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate average magnitude of motion vectors
        magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
        return magnitude

    except Exception:
        # Return 0 if motion estimation fails
        return 0.0


# -----------------------------------------------
# HELPER: Calculate driver safety score and
# build a list of violations based on detected
# signs and vehicle motion at each detection.
#
# Scoring:
#   Start at 100
#   -30 for not stopping at a stop sign
#   -40 for running a red light
#   -10 for possible speeding in a speed zone
# -----------------------------------------------
def calculate_score_and_violations(events):
    score = 100
    violations = []

    # Count unique signs encountered during the drive
    total_signs = len(set([e["sign"] for e in events]))

    for event in events:
        sign = event["sign"]
        motion = event["motion"]

        # Stop sign — vehicle should be stopped (motion ~0)
        if "stop" in sign.lower():
            if motion > 1.0:
                score -= 30
                violations.append("🛑 Did not stop at Stop Sign")

        # Red light — vehicle should be stopped (motion ~0)
        elif "red" in sign.lower():
            if motion > 1.0:
                score -= 40
                violations.append("🔴 Ran a Red Light")

        # Speed limit — vehicle should not be moving too fast
        elif any(str(n) in sign for n in [20, 30, 40, 50, 60, 70, 80, 90, 100]):
            if motion > 3.0:
                score -= 10
                violations.append(f"⚡ Possible Speeding in {sign} zone")

    # Clamp score to minimum of 0
    score = max(score, 0)
    return score, violations, total_signs


# -----------------------------------------------
# HELPER: Display the final drive summary,
# violations list, safety score, and speak
# the final verdict out loud.
# -----------------------------------------------
def show_final_result(score, violations, total_signs):
    st.markdown("---")
    st.markdown("## 📋 Drive Summary")
    st.write(
        f"The driver encountered **{total_signs} unique traffic signs** "
        f"and committed **{len(violations)} violation(s)**."
    )

    st.markdown("---")

    # Show violations if any, otherwise show success
    if violations:
        st.markdown("### ⚠️ Violations Detected")
        for v in violations:
            st.error(v)
    else:
        st.success("✅ No violations detected! Great driving.")

    st.markdown("---")

    # Show final safety score with colour coded badge
    st.markdown("### 🏆 Driver Safety Score")
    if score >= 80:
        st.success(f"✅ {score}/100 — Good Driver! Rules were respected.")
        speak("Analysis complete. Good driver. Rules were respected.")
    elif score >= 50:
        st.warning(
            f"⚠️ {score}/100 — Needs Improvement. Some violations detected.")
        speak("Analysis complete. The driver needs improvement. Some violations were detected.")
    else:
        st.error(
            f"🚨 {score}/100 — Dangerous Driving! Multiple violations detected.")
        speak(
            "Analysis complete. Dangerous driving detected. Multiple violations were found.")


# -----------------------------------------------
# MAIN APP — FILE UPLOAD & VIDEO PROCESSING
# -----------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a dashcam video",
    type=["mp4", "avi"]
)

if uploaded_file is not None:

    # Save uploaded video to a temporary file
    # so OpenCV can read it frame by frame
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(uploaded_file.read())

    # Show the uploaded video in the app
    st.video(uploaded_file)

    # Open video with OpenCV
    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(f"⏳ Analysing video... ({total_frames} frames total)")

    # UI elements for live feedback during processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    alert_box = st.empty()

    # Processing state variables
    frame_count = 0
    prev_frame = None
    last_spoken_sign = None  # tracks last spoken sign to avoid repetition
    events = []              # stores all sign detections with motion data

    # -----------------------------------------------
    # FRAME BY FRAME VIDEO PROCESSING LOOP
    # -----------------------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Update progress bar on every frame for smooth animation
        progress_bar.progress(min(frame_count / total_frames, 1.0))
        status_text.text(f"Analysing frame {frame_count} of {total_frames}...")

        # Skip frames to speed up processing
        # YOLO only runs every FRAME_SKIP frames
        if frame_count % FRAME_SKIP != 0:
            prev_frame = frame
            continue

        # Estimate how fast the vehicle is moving
        # using optical flow between current and previous frame
        motion = 0.0
        if prev_frame is not None:
            motion = estimate_motion(prev_frame, frame)

        # Run YOLO detection on current frame
        results = model(frame)

        for result in results:
            for box in result.boxes:

                # Skip low confidence detections
                confidence = float(box.conf)
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Get the label name for this detection
                label = model.names[int(box.cls)]

                # Store detection event for scoring later
                events.append({
                    "sign": label,
                    "motion": round(motion, 3)
                })

                # Trigger voice alert only when sign changes
                # This avoids repeating the same alert
                # while the same sign is still visible
                if label != last_spoken_sign:
                    message = get_alert_message(label)
                    if message:
                        alert_box.info(f"🔊 {message}")
                        speak(message)
                        last_spoken_sign = label

        prev_frame = frame

    # Clean up after processing
    cap.release()
    os.remove(tmp_path)

    # Clear live UI elements
    progress_bar.empty()
    status_text.empty()
    alert_box.empty()

    st.success(f"✅ Video analysed! ({frame_count} frames scanned)")

    # -----------------------------------------------
    # SHOW FINAL RESULTS
    # -----------------------------------------------
    if events:
        score, violations, total_signs = calculate_score_and_violations(events)
        show_final_result(score, violations, total_signs)
    else:
        st.warning("⚠️ No traffic signs detected in this footage.")
