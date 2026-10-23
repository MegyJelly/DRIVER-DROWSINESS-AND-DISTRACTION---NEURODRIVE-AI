import os
import glob
import argparse

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "face_landmarker.task"

EAR_THRESHOLD = 0.22
WINDOW_SECONDS = 1.5

SUPPORTED_EXTENSIONS = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp"
]


# ============================================================
# LANDMARKS
# ============================================================

LEFT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

RIGHT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


# ============================================================
# EAR
# ============================================================

def distance(a, b):

    return float(
        np.linalg.norm(
            np.array(a) -
            np.array(b)
        )
    )


def get_xy(
    landmarks,
    index,
    width,
    height
):

    point = landmarks[index]

    return np.array(
        [
            point.x * width,
            point.y * height
        ],
        dtype=np.float64
    )


def calculate_ear(
    landmarks,
    indices,
    width,
    height
):

    points = [
        get_xy(
            landmarks,
            index,
            width,
            height
        )
        for index in indices
    ]

    vertical_1 = distance(
        points[1],
        points[5]
    )

    vertical_2 = distance(
        points[2],
        points[4]
    )

    horizontal = distance(
        points[0],
        points[3]
    )

    if horizontal == 0:

        return 0.0

    return (
        vertical_1 +
        vertical_2
    ) / (
        2 *
        horizontal
    )


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    files = []

    for extension in SUPPORTED_EXTENSIONS:

        files.extend(
            glob.glob(
                os.path.join(
                    folder,
                    "**",
                    extension
                ),
                recursive=True
            )
        )

    return sorted(files)


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def get_frame_ear(
    image,
    landmarker
):

    height, width = image.shape[:2]

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(
        mp_image
    )

    if not result.face_landmarks:

        return None

    landmarks = result.face_landmarks[0]

    left = calculate_ear(
        landmarks,
        LEFT_EYE,
        width,
        height
    )

    right = calculate_ear(
        landmarks,
        RIGHT_EYE,
        width,
        height
    )

    return (
        left +
        right
    ) / 2.0


# ============================================================
# SLIDING WINDOW
# ============================================================

def classify_sequence(
    ear_values,
    fps
):

    if not ear_values:

        return 0

    required_frames = max(
        1,
        int(
            WINDOW_SECONDS *
            fps
        )
    )

    if len(ear_values) < required_frames:

        return 0

    for start in range(
        0,
        len(ear_values) -
        required_frames +
        1
    ):

        window = ear_values[
            start:
            start + required_frames
        ]

        closed_frames = sum(
            ear < EAR_THRESHOLD
            for ear in window
        )

        closed_ratio = (
            closed_frames /
            len(window)
        )

        if closed_ratio >= 0.90:

            return 1

    return 0


# ============================================================
# DATASET EVALUATION
# ============================================================

def evaluate_dataset(
    dataset_path,
    output_path
):

    drowsy_folder = os.path.join(
        dataset_path,
        "drowsy"
    )

    alert_folder = os.path.join(
        dataset_path,
        "notdrowsy"
    )

    if not os.path.isdir(
        drowsy_folder
    ):

        raise FileNotFoundError(
            "Missing dataset/drowsy folder."
        )

    if not os.path.isdir(
        alert_folder
    ):

        raise FileNotFoundError(
            "Missing dataset/notdrowsy folder."
        )


    drowsy_images = get_images(
        drowsy_folder
    )

    alert_images = get_images(
        alert_folder
    )

    print(
        f"Drowsy images: "
        f"{len(drowsy_images)}"
    )

    print(
        f"Non-drowsy images: "
        f"{len(alert_images)}"
    )


    BaseOptions = mp.tasks.BaseOptions

    FaceLandmarker = \
        mp.tasks.vision.FaceLandmarker

    FaceLandmarkerOptions = \
        mp.tasks.vision.FaceLandmarkerOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        num_faces=1
    )


    y_true = []
    y_pred = []

    processed = 0


    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:


        # ----------------------------------------------------
        # DROWSY
        # ----------------------------------------------------

        for image_path in drowsy_images:

            image = cv2.imread(
                image_path
            )

            if image is None:
                continue

            ear = get_frame_ear(
                image,
                landmarker
            )

            if ear is None:
                continue

            # For image-level evaluation,
            # each frame is evaluated as a labelled sample.
            prediction = int(
                ear < EAR_THRESHOLD
            )

            y_true.append(1)
            y_pred.append(prediction)

            processed += 1


        # ----------------------------------------------------
        # NON-DROWSY
        # ----------------------------------------------------

        for image_path in alert_images:

            image = cv2.imread(
                image_path
            )

            if image is None:
                continue

            ear = get_frame_ear(
                image,
                landmarker
            )

            if ear is None:
                continue

            prediction = int(
                ear < EAR_THRESHOLD
            )

            y_true.append(0)
            y_pred.append(prediction)

            processed += 1


    if not y_true:

        raise RuntimeError(
            "No valid facial samples were processed."
        )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    print()
    print(
        "================================"
    )

    print(
        "NEURODRIVE AI EVALUATION"
    )

    print(
        "================================"
    )

    print(
        f"Samples processed: {processed}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Not Drowsy",
            "Drowsy"
        ],
        zero_division=0
    )

    print()
    print(report)


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_true,
        y_pred
    )


    print(
        "Confusion Matrix:"
    )

    print(matrix)


    # ========================================================
    # FALSE POSITIVE / FALSE NEGATIVE
    # ========================================================

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = (
        fp /
        (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn /
        (fn + tp)
        if (fn + tp) > 0
        else 0
    )


    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

    print(
        f"False Negative Rate: "
        f"{false_negative_rate:.4f}"
    )


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    os.makedirs(
        output_path,
        exist_ok=True
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "NeuroDrive AI - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        [0, 1],
        [
            "Not Drowsy",
            "Drowsy"
        ]
    )

    plt.yticks(
        [0, 1],
        [
            "Not Drowsy",
            "Drowsy"
        ]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    matrix_path = os.path.join(
        output_path,
        "confusion_matrix.png"
    )

    plt.savefig(
        matrix_path,
        dpi=200
    )

    plt.close()


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = pd.DataFrame(
        [
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "False Positive Rate":
                    false_positive_rate,
                "False Negative Rate":
                    false_negative_rate
            }
        ]
    )

    metrics.to_csv(
        os.path.join(
            output_path,
            "evaluation_metrics.csv"
        ),
        index=False
    )


    print()
    print(
        f"Saved report to: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="dataset"
    )

    parser.add_argument(
        "--output",
        default="evaluation_results"
    )

    args = parser.parse_args()

    evaluate_dataset(
        args.dataset,
        args.output
    )
