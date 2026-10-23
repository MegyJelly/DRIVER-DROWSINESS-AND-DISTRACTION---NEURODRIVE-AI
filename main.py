
import os
import time
import threading
from datetime import datetime
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

try:
    import winsound
except ImportError:
    winsound = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroDrive AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "face_landmarker.task"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&display=swap');

:root {
    --sage: #dcebd4;
    --deep-green: #426b45;
    --pink: #e9a9bd;
    --deep-pink: #a95473;
}

.stApp {
    background: linear-gradient(
        135deg,
        #b7e887 0%,
        #a2e362 38%,
        #ceedaf 70%,
        #fac5e4 100%
    );
    font-family: 'Rockwell', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #426b45 !important;
    font-family: 'Rockwell', sans-serif !important;
}

p, label, .stMarkdown {
    font-family: 'Rockwell', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #d0e6c7 0%,
        #e1efd9 65%,
        #f9e4ed 100%
    );
    border-right: 1px solid #d5e3ce;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #386440 !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] {
    color: #527456;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #f9fff6,
        #fff0f5
    );
    border: 1px solid #e7d2dc;
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 4px 14px rgba(75, 110, 75, 0.07);
}

[data-testid="stMetricLabel"] {
    color: #527456;
    font-weight: 700;
}

[data-testid="stMetricValue"] {
    color: #426b45;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        135deg,
        #d4e8ca,
        #f7dce6
    );
    color: #426b45;
    border: 1px solid #e6c5d2;
    border-radius: 13px;
    font-weight: 800;
    min-height: 43px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #f2c7d8;
    color: #87445f;
    border-color: #e9a9bd;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #a95473;
    border-bottom-color: #e9a9bd;
}

.nd-hero {
    background: linear-gradient(
        115deg,
        #d7eacc,
        #fff0f5
    );
    border: 1px solid #dce5d5;
    border-radius: 24px;
    padding: 24px 28px;
    margin-bottom: 20px;
}

.nd-subtitle {
    color: #58765b;
    font-size: 15px;
    margin-top: -7px;
}

.nd-panel {
    background: rgba(255, 255, 255, 0.76);
    border: 1px solid #e9dbe0;
    border-radius: 18px;
    padding: 17px;
}

.nd-footer {
    text-align: center;
    color: #58765b;
    font-size: 12px;
    padding: 24px 0 10px;
}

hr {
    border-color: #d7e5d0;
}


/* Soft botanical panels */
.nd-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        135deg,
        #bdf79c,
        #fad4e3
    );
    border: 1px solid #49b80d;
    border-radius: 26px;
    padding: 28px;
    box-shadow: 0 6px 20px rgba(82, 116, 86, 0.08);
}

/* Small floral heading accent */
.nd-hero h1::after {
    content: "  ✿";
    color: #d58da7;
    font-size: 0.7em;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 250, 248, 0.88);
    border: 1px solid #e7cdd8;
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 5px 16px rgba(82, 116, 86, 0.07);
    transition: transform 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
}

/* Soft pink buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 16px;
    border: 1px solid #e5bfd0;
    background: linear-gradient(
        135deg,
        #d9ebd0,
        #f8dce7
    );
    color: #426b45;
    font-weight: 800;
    box-shadow: 0 3px 9px rgba(169, 84, 115, 0.07);
}

/* Pastel green section headings */
h2, h3 {
    color: #426b45 !important;
}

/* Subtle pink dividers */
hr {
    border-color: #fcbbf4 !important;
}

/* Rounded camera preview */
[data-testid="stImage"] img {
    border: 2px solid #e6c9d5;
    border-radius: 18px;
}

/* Sidebar slider accent */
[data-testid="stSidebar"] [data-testid="stSlider"] {
    accent-color: #2f6130;
}

/* Footer decoration */
.nd-footer::before {
    content: "✿  ";
    color: #d4668d;
}

.nd-footer::after {
    content: "  ✿";
    color: #d97e9f;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="nd-hero">
    <h1>NeuroDrive AI <span style="color:#d58da7;">✿</span></h1>
    <p class="nd-subtitle">
        AI-assisted driver monitoring for detecting signs of
        drowsiness and distraction.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "running": False,
    "events": [],
    "blinks": 0,
    "blink_times": deque(),
    "yawns": 0,
    "distractions": 0,
    "drowsiness_events": 0,
    "last_eye_closed": False,
    "last_yawn": False,
    "last_distraction": False,
    "last_drowsiness": False,
    "eye_closed_since": None,
    "distraction_since": None,
    "alarm_active": False,
    "frame_count": 0,
    "fps": 0.0,
    "camera": None,
    "ear_history": deque(maxlen=120),
    "mar_history": deque(maxlen=120),
    "neutral_yaw": None,
    "neutral_pitch": None,
    "calibration_samples": [],
    "calibration_complete": False,
    "alarm_lock": threading.Lock(),
    "session_started": datetime.now()
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.markdown("## Detection Settings")
st.sidebar.caption("Adjust the sensitivity of the system.")

ear_threshold = st.sidebar.slider(
    "Eye closure sensitivity (EAR)",
    min_value=0.10,
    max_value=0.35,
    value=0.21,
    step=0.01
)

mar_threshold = st.sidebar.slider(
    "Yawn sensitivity (MAR)",
    min_value=0.20,
    max_value=0.90,
    value=0.55,
    step=0.05
)

eye_closure_seconds = st.sidebar.slider(
    "Drowsiness warning delay (seconds)",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1
)

yaw_threshold = st.sidebar.slider(
    "Head-turn threshold (degrees)",
    min_value=10,
    max_value=45,
    value=25
)

pitch_threshold = st.sidebar.slider(
    "Head-tilt threshold (degrees)",
    min_value=5,
    max_value=40,
    value=12
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Detection guide")
st.sidebar.caption("EAR: Eye Aspect Ratio")
st.sidebar.caption("MAR: Mouth Aspect Ratio")
st.sidebar.caption("Yaw: Horizontal head rotation")
st.sidebar.caption("Pitch: Vertical head rotation")
st.sidebar.caption("Blinks/min: Blinks in the last 60 seconds")


# ============================================================
# FACIAL MEASUREMENTS
# ============================================================

def euclidean_distance(point1, point2):
    return float(
        np.linalg.norm(
            np.asarray(point1) - np.asarray(point2)
        )
    )


def calculate_ear(landmarks, eye_indices):
    points = [
        (landmarks[index].x, landmarks[index].y)
        for index in eye_indices
    ]

    vertical_1 = euclidean_distance(points[1], points[5])
    vertical_2 = euclidean_distance(points[2], points[4])
    horizontal = euclidean_distance(points[0], points[3])

    if horizontal < 1e-8:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks):
    points = [
        (landmarks[index].x, landmarks[index].y)
        for index in MOUTH
    ]

    vertical_1 = euclidean_distance(points[1], points[7])
    vertical_2 = euclidean_distance(points[2], points[6])
    vertical_3 = euclidean_distance(points[3], points[5])
    horizontal = euclidean_distance(points[0], points[4])

    if horizontal < 1e-8:
        return 0.0

    return (
        vertical_1 + vertical_2 + vertical_3
    ) / (3.0 * horizontal)


def calculate_head_pose(landmarks, width, height):
    image_points = np.array([
        [landmarks[1].x * width, landmarks[1].y * height],
        [landmarks[152].x * width, landmarks[152].y * height],
        [landmarks[33].x * width, landmarks[33].y * height],
        [landmarks[263].x * width, landmarks[263].y * height],
        [landmarks[61].x * width, landmarks[61].y * height],
        [landmarks[291].x * width, landmarks[291].y * height]
    ], dtype=np.float64)

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0]
    ], dtype=np.float64)

    focal_length = float(width)
    center = (width / 2.0, height / 2.0)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    pitch = float(angles[0])
    yaw = float(angles[1])

    return yaw, pitch


def calculate_blinks_per_minute():
    current_time = time.time()
    blink_times = st.session_state.blink_times

    while (
        blink_times
        and current_time - blink_times[0] > 60
    ):
        blink_times.popleft()

    return len(blink_times)


# ============================================================
# EVENT LOGGING
# ============================================================

def log_event(event_type, details=""):
    now = datetime.now()
    events = st.session_state.events

    if events:
        previous = events[-1]

        if (
            previous["Event"] == event_type
            and (now - previous["_datetime"]).total_seconds() < 5
        ):
            return

    events.append({
        "Time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Event": event_type,
        "Details": details,
        "_datetime": now
    })

    if len(events) > 500:
        del events[:-500]


# ============================================================
# ALARM
# ============================================================

def play_two_beeps():
    if winsound is None:
        return

    try:
        winsound.Beep(1200, 250)
        time.sleep(0.12)
        winsound.Beep(1200, 250)
    except Exception:
        pass


def trigger_alarm_once():
    with st.session_state.alarm_lock:
        if st.session_state.alarm_active:
            return

        st.session_state.alarm_active = True

    threading.Thread(
        target=play_two_beeps,
        daemon=True
    ).start()


def reset_alarm():
    with st.session_state.alarm_lock:
        st.session_state.alarm_active = False


# ============================================================
# RESET
# ============================================================

def reset_session():
    st.session_state.events = []
    st.session_state.blinks = 0
    st.session_state.blink_times.clear()
    st.session_state.yawns = 0
    st.session_state.distractions = 0
    st.session_state.drowsiness_events = 0

    st.session_state.last_eye_closed = False
    st.session_state.last_yawn = False
    st.session_state.last_distraction = False
    st.session_state.last_drowsiness = False

    st.session_state.eye_closed_since = None
    st.session_state.distraction_since = None

    st.session_state.neutral_yaw = None
    st.session_state.neutral_pitch = None
    st.session_state.calibration_samples = []
    st.session_state.calibration_complete = False

    st.session_state.frame_count = 0
    st.session_state.fps = 0.0

    st.session_state.ear_history.clear()
    st.session_state.mar_history.clear()

    reset_alarm()
    st.session_state.session_started = datetime.now()


# ============================================================
# START / STOP CONTROLS
# ============================================================

button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    start_button = st.button(
        "Start monitoring",
        use_container_width=True
    )

with button_col2:
    stop_button = st.button(
        "Stop monitoring",
        use_container_width=True
    )

with button_col3:
    reset_button = st.button(
        "Reset session",
        use_container_width=True
    )


if reset_button:
    st.session_state.running = False

    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None

    reset_session()
    st.rerun()


if start_button:
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Missing face_landmarker.task. "
            "Place it in the same folder as main.py."
        )
    else:
        if st.session_state.camera is None:
            camera = cv2.VideoCapture(0)

            if camera.isOpened():
                st.session_state.camera = camera
                reset_session()
                st.session_state.running = True
            else:
                camera.release()
                st.error(
                    "Could not open the webcam. Check camera permissions "
                    "and close other applications using it."
                )
        else:
            st.session_state.running = True


if stop_button:
    st.session_state.running = False

    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None

    reset_alarm()
    st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

status_box = st.empty()
video_placeholder = st.empty()

metric_cols = st.columns(6)

blink_metric = metric_cols[0].empty()
blink_rate_metric = metric_cols[1].empty()
yawn_metric = metric_cols[2].empty()
distraction_metric = metric_cols[3].empty()
drowsiness_metric = metric_cols[4].empty()
fps_metric = metric_cols[5].empty()

chart_col1, chart_col2 = st.columns(2)

ear_chart = chart_col1.empty()
mar_chart = chart_col2.empty()

st.markdown("### Safety event overview")
memory_placeholder = st.empty()

st.markdown("### Recent safety events")
event_placeholder = st.empty()


def update_metrics():
    blink_metric.metric(
        "Total blinks",
        st.session_state.blinks
    )

    blink_rate_metric.metric(
        "Blinks/min",
        calculate_blinks_per_minute()
    )

    yawn_metric.metric(
        "Yawns",
        st.session_state.yawns
    )

    distraction_metric.metric(
        "Distractions",
        st.session_state.distractions
    )

    drowsiness_metric.metric(
        "Drowsiness events",
        st.session_state.drowsiness_events
    )

    fps_metric.metric(
        "FPS",
        f"{st.session_state.fps:.1f}"
    )


def update_event_dashboard():
    events = st.session_state.events

    if not events:
        memory_placeholder.info(
            "Safety-event charts will appear after events are recorded."
        )
        event_placeholder.info("No safety events recorded yet.")
        return

    event_df = pd.DataFrame([
        {
            "Time": event["Time"],
            "Event": event["Event"],
            "Details": event["Details"],
            "_datetime": event["_datetime"]
        }
        for event in events
    ])

    event_df["Minute"] = event_df["_datetime"].dt.strftime("%H:%M")

    memory_df = (
        event_df.groupby(["Minute", "Event"])
        .size()
        .unstack(fill_value=0)
    )

    memory_placeholder.bar_chart(memory_df, height=220)

    event_placeholder.dataframe(
        event_df[["Time", "Event", "Details"]]
        .tail(15)
        .iloc[::-1],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# LIVE MONITORING
# ============================================================

if st.session_state.running:
    if not os.path.exists(MODEL_PATH):
        st.error("Missing face_landmarker.task model file.")
        st.session_state.running = False

    else:
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        try:
            with mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            ) as landmarker:

                previous_frame_time = time.time()
                last_timestamp_ms = 0

                status_box.info(
                    "Calibration: look straight at the camera and "
                    "hold your head still for about two seconds."
                )

                while st.session_state.running:
                    camera = st.session_state.camera

                    if camera is None:
                        break

                    success, frame = camera.read()

                    if not success:
                        status_box.warning("Unable to read from webcam.")
                        break

                    frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    timestamp_ms = max(
                        int(time.time() * 1000),
                        last_timestamp_ms + 1
                    )
                    last_timestamp_ms = timestamp_ms

                    result = landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )

                    now = time.time()
                    elapsed = now - previous_frame_time
                    previous_frame_time = now

                    if elapsed > 0:
                        st.session_state.fps = 1.0 / elapsed

                    st.session_state.frame_count += 1

                    ear = 0.0
                    mar = 0.0
                    yaw = 0.0
                    pitch = 0.0

                    eyes_closed = False
                    yawning = False
                    distracted = False
                    drowsy = False

                    calibration_complete = (
                        st.session_state.calibration_complete
                    )

                    if result.face_landmarks:
                        landmarks = result.face_landmarks[0]

                        ear_left = calculate_ear(
                            landmarks,
                            LEFT_EYE
                        )
                        ear_right = calculate_ear(
                            landmarks,
                            RIGHT_EYE
                        )

                        ear = (ear_left + ear_right) / 2.0
                        mar = calculate_mar(landmarks)

                        yaw, pitch = calculate_head_pose(
                            landmarks,
                            width,
                            height
                        )

                        eyes_closed = ear < ear_threshold
                        yawning = mar > mar_threshold

                        # ------------------------------------------------
                        # CALIBRATION
                        # ------------------------------------------------

                        if not st.session_state.calibration_complete:
                            # Only collect calibration samples while the
                            # eyes appear open and the face is detected.
                            if not eyes_closed:
                                st.session_state.calibration_samples.append(
                                    (yaw, pitch)
                                )

                            if (
                                len(st.session_state.calibration_samples)
                                >= 45
                            ):
                                samples = np.asarray(
                                    st.session_state.calibration_samples,
                                    dtype=np.float64
                                )

                                st.session_state.neutral_yaw = float(
                                    np.median(samples[:, 0])
                                )
                                st.session_state.neutral_pitch = float(
                                    np.median(samples[:, 1])
                                )

                                st.session_state.calibration_complete = True

                            calibration_complete = (
                                st.session_state.calibration_complete
                            )

                        # ------------------------------------------------
                        # CALIBRATED HEAD-POSE DETECTION
                        # ------------------------------------------------

                        if calibration_complete:
                            yaw_deviation = abs(
                                yaw - st.session_state.neutral_yaw
                            )

                            pitch_deviation = abs(
                                pitch - st.session_state.neutral_pitch
                            )

                            head_pose_out_of_range = (
                                yaw_deviation > yaw_threshold
                                or pitch_deviation > pitch_threshold
                            )

                            if head_pose_out_of_range:
                                if (
                                    st.session_state.distraction_since
                                    is None
                                ):
                                    st.session_state.distraction_since = now

                                distracted = (
                                    now
                                    - st.session_state.distraction_since
                                    >= 0.8
                                )
                            else:
                                st.session_state.distraction_since = None
                                distracted = False

                        # ------------------------------------------------
                        # SUSTAINED EYE CLOSURE
                        # ------------------------------------------------

                        if eyes_closed:
                            if st.session_state.eye_closed_since is None:
                                st.session_state.eye_closed_since = now

                            closure_duration = (
                                now - st.session_state.eye_closed_since
                            )

                            drowsy = (
                                closure_duration >= eye_closure_seconds
                            )
                        else:
                            st.session_state.eye_closed_since = None
                            drowsy = False

                        # ------------------------------------------------
                        # BLINK COUNTING
                        # ------------------------------------------------

                        if (
                            not eyes_closed
                            and st.session_state.last_eye_closed
                        ):
                            st.session_state.blinks += 1
                            st.session_state.blink_times.append(now)
                            log_event("Blink detected")

                        if (
                            eyes_closed
                            and not st.session_state.last_eye_closed
                        ):
                            log_event(
                                "Eyes closed",
                                f"EAR: {ear:.3f}"
                            )

                        # ------------------------------------------------
                        # YAWN COUNTING
                        # ------------------------------------------------

                        if (
                            yawning
                            and not st.session_state.last_yawn
                        ):
                            st.session_state.yawns += 1
                            log_event(
                                "Yawn detected",
                                f"MAR: {mar:.3f}"
                            )

                        # ------------------------------------------------
                        # DISTRACTION EVENTS
                        # ------------------------------------------------

                        if (
                            distracted
                            and not st.session_state.last_distraction
                        ):
                            st.session_state.distractions += 1
                            log_event(
                                "Head pose deviation",
                                f"Yaw deviation: {yaw_deviation:.1f}°, "
                                f"pitch deviation: {pitch_deviation:.1f}°"
                            )

                        # ------------------------------------------------
                        # DROWSINESS EVENTS
                        # ------------------------------------------------

                        if (
                            drowsy
                            and not st.session_state.last_drowsiness
                        ):
                            st.session_state.drowsiness_events += 1
                            log_event(
                                "Drowsiness warning",
                                f"Eyes closed for at least "
                                f"{eye_closure_seconds:.1f} seconds"
                            )

                        st.session_state.last_eye_closed = eyes_closed
                        st.session_state.last_yawn = yawning
                        st.session_state.last_distraction = distracted
                        st.session_state.last_drowsiness = drowsy

                        # ------------------------------------------------
                        # ALARM
                        # ------------------------------------------------

                        if drowsy or distracted:
                            trigger_alarm_once()
                        else:
                            reset_alarm()

                        # ------------------------------------------------
                        # CAMERA OVERLAY
                        # ------------------------------------------------

                        for point in landmarks:
                            px = int(point.x * width)
                            py = int(point.y * height)

                            cv2.circle(
                                frame,
                                (px, py),
                                1,
                                (170, 210, 175),
                                -1
                            )

                        if not calibration_complete:
                            status_text = "CALIBRATING - LOOK STRAIGHT AHEAD"
                            status_color = (90, 130, 90)
                        elif drowsy:
                            status_text = "DROWSINESS DETECTED"
                            status_color = (60, 60, 220)
                        elif distracted:
                            status_text = "HEAD MOVEMENT DETECTED"
                            status_color = (170, 90, 180)
                        elif yawning:
                            status_text = "POSSIBLE YAWN"
                            status_color = (180, 130, 190)
                        else:
                            status_text = "FORWARD / NORMAL POSITION"
                            status_color = (70, 145, 75)

                        cv2.rectangle(
                            frame,
                            (10, 10),
                            (width - 10, 55),
                            (250, 255, 248),
                            -1
                        )

                        cv2.putText(
                            frame,
                            status_text,
                            (20, 39),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            status_color,
                            2,
                            cv2.LINE_AA
                        )

                        cv2.putText(
                            frame,
                            f"EAR: {ear:.3f}   MAR: {mar:.3f}",
                            (20, height - 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (65, 110, 70),
                            2,
                            cv2.LINE_AA
                        )

                        cv2.putText(
                            frame,
                            f"Yaw: {yaw:.1f}   Pitch: {pitch:.1f}",
                            (20, height - 42),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (65, 110, 70),
                            2,
                            cv2.LINE_AA
                        )

                        if calibration_complete:
                            cv2.putText(
                                frame,
                                f"Yaw change: {yaw_deviation:.1f}   "
                                f"Pitch change: {pitch_deviation:.1f}",
                                (20, height - 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.48,
                                (65, 110, 70),
                                1,
                                cv2.LINE_AA
                            )

                    else:
                        # Do not classify a missing face as attentive.
                        reset_alarm()

                        st.session_state.eye_closed_since = None
                        st.session_state.distraction_since = None

                        st.session_state.last_eye_closed = False
                        st.session_state.last_yawn = False
                        st.session_state.last_distraction = False
                        st.session_state.last_drowsiness = False

                        cv2.putText(
                            frame,
                            "FACE NOT DETECTED",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (150, 80, 170),
                            2,
                            cv2.LINE_AA
                        )

                    # ----------------------------------------------------
                    # UPDATE DASHBOARD
                    # ----------------------------------------------------

                    st.session_state.ear_history.append(ear)
                    st.session_state.mar_history.append(mar)

                    video_placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )

                    update_metrics()

                    if not st.session_state.calibration_complete:
                        status_box.warning(
                            "Calibrating: look straight at the camera "
                            "and keep your head still."
                        )
                    else:
                        status_box.success(
                            "Calibration complete. Monitoring is active."
                        )

                    ear_chart.line_chart(
                        pd.DataFrame({
                            "EAR": list(st.session_state.ear_history)
                        }),
                        height=200
                    )

                    mar_chart.line_chart(
                        pd.DataFrame({
                            "MAR": list(st.session_state.mar_history)
                        }),
                        height=200
                    )

                    update_event_dashboard()

                    time.sleep(0.01)

        except Exception as error:
            st.error(f"Monitoring error: {error}")

        finally:
            if st.session_state.camera is not None:
                st.session_state.camera.release()
                st.session_state.camera = None

            st.session_state.running = False
            reset_alarm()

            status_box.info(
                "Monitoring stopped. Select Start monitoring to begin again."
            )

else:
    status_box.info(
        "Monitoring is stopped. Select Start monitoring to activate "
        "the webcam."
    )

    update_metrics()
    update_event_dashboard()

    if st.session_state.ear_history:
        ear_chart.line_chart(
            pd.DataFrame({
                "EAR": list(st.session_state.ear_history)
            }),
            height=200
        )

        mar_chart.line_chart(
            pd.DataFrame({
                "MAR": list(st.session_state.mar_history)
            }),
            height=200
        )


# ============================================================
# CSV REPORT
# ============================================================

st.markdown("### Export safety report")

if st.session_state.events:
    export_df = pd.DataFrame([
        {
            "Time": event["Time"],
            "Event": event["Event"],
            "Details": event["Details"]
        }
        for event in st.session_state.events
    ])

    st.download_button(
        label="Download safety report (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="neurodrive_safety_report.csv",
        mime="text/csv"
    )
else:
    st.caption("Your report will be available after events are recorded.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="nd-footer">
    NeuroDrive AI | AI-assisted driver monitoring prototype
    <br>
    For demonstration and testing only. Not a certified vehicle
    safety system.
</div>
""", unsafe_allow_html=True)
