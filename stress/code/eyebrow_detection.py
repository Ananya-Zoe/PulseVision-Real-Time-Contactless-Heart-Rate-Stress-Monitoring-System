import streamlit as st
import os
import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance as dist
from imutils import face_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from blink_detection import BlinkDetector
from mouth_tension import MouthTensionDetector

st.set_page_config(page_title="Stress Measurement", page_icon="📈")

# Load and cache models for performance
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
    )
    emotion_classifier = load_model(
        os.path.join(os.path.dirname(__file__), "_mini_XCEPTION.102-0.66.hdf5"), compile=False
    )
    return detector, predictor, emotion_classifier

detector, predictor, emotion_classifier = load_models()

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
# Constants
MEASUREMENT_DURATION = 60  # seconds
WEIGHTS = {
    'eyebrow': 0.4,
    'blink': 0.25,
    'mouth': 0.2,
    'emotion': 0.15
}

st.title("Stress Measurement Using Face Recognition")
st.write("Detects stress via facial cues: eyebrow distance, blinking, mouth tension, and emotion.")
st.info(f"Stay still and in a well-lit area. Measurement runs for {MEASUREMENT_DURATION} seconds.")

start_button = st.button("Start Measurement")
stop_button = st.button("Stop Measuring")

# Session state
if "measuring" not in st.session_state:
    st.session_state.measuring = False
if "last_stress_value" not in st.session_state:
    st.session_state.last_stress_value = None
    st.session_state.last_stress_label = None
if "stress_values" not in st.session_state:
    st.session_state.stress_values = []  # List to store individual stress values

if start_button:
    st.session_state.measuring = True
    st.session_state.start_time = time.time()
if stop_button:
    st.session_state.measuring = False

frame_placeholder = st.empty()
stress_placeholder = st.empty()
timer_placeholder = st.empty()
progress_bar = st.progress(0)

# Helper functions
def eye_brow_distance(leye, reye):
    return dist.euclidean(leye, reye)

def emotion_score(label):
    return 1.0 if label == 'stressed' else 0.0

def emotion_finder(face, frame):
    x, y, w, h = face_utils.rect_to_bb(face)
    roi = cv2.resize(frame[y:y+h, x:x+w], (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_classifier.predict(roi)[0]
    label = EMOTIONS[preds.argmax()]
    return label

def normalize_eyebrow_stress(points, current):
    if len(points) < 2:
        return 0
    norm_val = abs(current - min(points)) / (max(points) - min(points))
    return np.exp(-norm_val)

def blink_stress_score(blinks, duration_sec):
    bpm = blinks / (duration_sec / 60) if duration_sec > 0 else 0
    if bpm < 10:
        return 1.0
    elif 10 <= bpm <= 25:
        return 0.3
    return 0.8

# Video and detectors
cap = cv2.VideoCapture(0)
blink_detector = BlinkDetector(ear_threshold=0.21, consec_frames=1)
mouth_detector = MouthTensionDetector()
points = []

# Main loop
if st.session_state.measuring:
    try:
        while cap.isOpened() and st.session_state.measuring:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            elapsed = time.time() - st.session_state.start_time
            if elapsed >= MEASUREMENT_DURATION:
                st.session_state.measuring = False
            prog = min(elapsed / MEASUREMENT_DURATION, 1.0)
            progress_bar.progress(prog)
            timer_placeholder.write(f"Time: {int(elapsed)}s / {MEASUREMENT_DURATION}s")

            for face in faces:
                emotion_label = emotion_finder(face, gray)
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                l_i, l_e = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']
                r_i, r_e = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']
                leyebrow = shape[l_i:l_e]
                reyebrow = shape[r_i:r_e]

                d = eye_brow_distance(leyebrow[-1], reyebrow[0])
                points.append(int(d))
                eb = normalize_eyebrow_stress(points, d)
                ear, blinks = blink_detector.update(shape)
                bl = blink_stress_score(blinks, elapsed)
                mt = mouth_detector.get_tension_score(shape)
                es = emotion_score(emotion_label)

                total = (
                    WEIGHTS['eyebrow'] * eb +
                    WEIGHTS['blink'] * bl +
                    WEIGHTS['mouth'] * mt +
                    WEIGHTS['emotion'] * es
                )
                label = 'High Stress' if total >= 0.60 else 'Low Stress'

                st.session_state.last_stress_value = f"{int(total*100)}%"
                st.session_state.last_stress_label = label

                # Store the stress value for average calculation
                st.session_state.stress_values.append(total)

                # Draw annotations
                x, y, w, h = face_utils.rect_to_bb(face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                for pt in np.vstack((leyebrow, reyebrow)):
                    cv2.circle(frame, tuple(pt), 2, (255, 0, 0), -1)
                cv2.putText(frame, f"Stress: {int(total*100)}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion_label}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"EAR: {ear:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Blinks: {blinks}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"MouthTension: {mt:.2f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            frame_placeholder.image(frame, channels="BGR")
            stress_placeholder.write(f"Current Stress: {st.session_state.last_stress_label}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
else:
    cap.release()
    cv2.destroyAllWindows()

# Final results
if st.session_state.stress_values:
    # Calculate the average stress value
    average_stress = np.mean(st.session_state.stress_values)

    # Check if the average stress is NaN
    if np.isnan(average_stress):
        average_stress = 0  # Default to 0 if NaN

    # Display the average stress value
    st.markdown("---")
    st.subheader("Final Monitoring Results")
    st.write(f"*Average Stress Level:* {int(average_stress * 100)}%")
    st.write(f"*Stress Status:* {'High Stress' if average_stress >= 0.65 else 'Low Stress'}")
else:
    st.write("No stress data available.")