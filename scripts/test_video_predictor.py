import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


# ============================================================
# Repository setup
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.fdmse_isl_spoter import FDMSEISLSPOTER


# ============================================================
# Configuration
# ============================================================

MANIFEST = ROOT / "manifests" / "landmark_dataset_index.csv"

CHECKPOINT = (
    ROOT
    / "models"
    / "checkpoints"
    / "best_fdmse_isl_spoter.pth"
)

NUM_CLASSES = 53
INPUT_DIM = 177
HIDDEN_DIM = 108

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load test manifest
# ============================================================

df = pd.read_csv(MANIFEST)

test_df = (
    df[df["split"] == "test"]
    .reset_index(drop=True)
)


# ============================================================
# Class mapping
# ============================================================

class_mapping = (
    test_df[
        ["label", "class"]
    ]
    .drop_duplicates()
    .sort_values("label")
)

label_to_class = dict(
    zip(
        class_mapping["label"],
        class_mapping["class"]
    )
)

class_to_label = {
    name: int(label)
    for label, name in label_to_class.items()
}


# ============================================================
# Load model
# ============================================================

print("=" * 80)
print("FDMSE-ISL TEST VIDEO PREDICTOR")
print("=" * 80)

print("Loading model...")

model = FDMSEISLSPOTER(
    num_classes=NUM_CLASSES,
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Validation accuracy:",
    f"{checkpoint['val_accuracy']:.2f}%"
)

print(
    "Device:",
    DEVICE
)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(landmark_path):

    landmarks = np.load(
        landmark_path
    )

    if landmarks.ndim != 2:
        raise ValueError(
            f"Expected shape (T,177), "
            f"got {landmarks.shape}"
        )

    if landmarks.shape[1] != 177:
        raise ValueError(
            f"Expected 177 features/frame, "
            f"got {landmarks.shape[1]}"
        )

    x = torch.from_numpy(
        landmarks.astype(np.float32)
    ).to(DEVICE)

    logits = model(x)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidence, prediction = (
        probabilities.max(dim=1)
    )

    return (
        prediction.item(),
        confidence.item() * 100.0
    )


# ============================================================
# Play video
# ============================================================

def show_video(video_path, ground_truth, prediction):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print()
        print(
            "WARNING: Could not open video:"
        )

        print(video_path)

        return

    window_title = (
        f"Ground Truth: {ground_truth} | "
        f"Prediction: {prediction}"
    )

    print()
    print(
        "Playing video..."
    )

    print(
        "Press Q inside the video window to stop."
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow(
            window_title,
            frame
        )

        key = cv2.waitKey(15) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Display signs
# ============================================================

def choose_sign():

    print()
    print("=" * 80)
    print("AVAILABLE TEST SIGNS")
    print("=" * 80)

    for label in sorted(label_to_class):

        print(
            f"{label + 1:2d}. "
            f"{label_to_class[label]}"
        )

    print()
    print(" 0. Quit")

    while True:

        value = input(
            "\nEnter sign number: "
        ).strip()

        if value == "0":
            return None

        try:
            number = int(value)

        except ValueError:

            print(
                "Please enter a valid number."
            )

            continue

        label = number - 1

        if label in label_to_class:
            return label

        print(
            "Invalid sign number."
        )


# ============================================================
# Choose signer/video
# ============================================================

def choose_video(label):

    class_df = (
        test_df[
            test_df["label"] == label
        ]
        .reset_index(drop=True)
    )

    class_name = label_to_class[label]

    print()
    print("=" * 80)
    print(
        f"TEST VIDEOS FOR: {class_name}"
    )
    print("=" * 80)

    for i, row in class_df.iterrows():

        print(
            f"{i + 1}. "
            f"Signer: {row['signer']:<6} "
            f"Video: {row['video_name']}"
        )

    print()
    print(" 0. Back")

    while True:

        value = input(
            "\nChoose video/signer: "
        ).strip()

        if value == "0":
            return None

        try:
            number = int(value)

        except ValueError:

            print(
                "Please enter a valid number."
            )

            continue

        if 1 <= number <= len(class_df):

            return class_df.iloc[
                number - 1
            ]

        print(
            f"Please enter 1-{len(class_df)}."
        )


# ============================================================
# Main
# ============================================================

def main():

    while True:

        # ----------------------------------------------------
        # Choose sign
        # ----------------------------------------------------

        label = choose_sign()

        if label is None:
            break

        # ----------------------------------------------------
        # Choose test video
        # ----------------------------------------------------

        row = choose_video(label)

        if row is None:
            continue

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        video_name = row["video_name"]
        signer = row["signer"]
        true_class = row["class"]
        true_label = int(row["label"])

        landmark_path = (
            ROOT
            / row["landmarks_177_path"]
        )

        # ----------------------------------------------------
        # Read metadata
        # ----------------------------------------------------

        video_landmark_dir = (
            ROOT
            / row["landmarks_path"]
        ).parent

        metadata_path = (
            video_landmark_dir
            / "metadata.json"
        )

        if not metadata_path.exists():

            print(
                "\nERROR: metadata.json not found."
            )

            continue

        with open(
            metadata_path,
            "r"
        ) as f:

            metadata = json.load(f)

        original_video_path = Path(
            metadata["video_path"]
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction_label, confidence = predict(
            landmark_path
        )

        prediction_class = label_to_class[
            prediction_label
        ]

        correct = (
            prediction_label == true_label
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("PREDICTION RESULT")
        print("=" * 80)

        print(
            "Video:",
            video_name
        )

        print(
            "Signer:",
            signer
        )

        print(
            "Ground truth:",
            true_class
        )

        print()
        print(
            "MODEL PREDICTION:",
            prediction_class
        )

        print(
            "Confidence:",
            f"{confidence:.2f}%"
        )

        print()

        if correct:

            print(
                "RESULT: CORRECT"
            )

        else:

            print(
                "RESULT: WRONG"
            )

        print("=" * 80)

        # ----------------------------------------------------
        # Video
        # ----------------------------------------------------

        if original_video_path.exists():

            show_video(
                original_video_path,
                true_class,
                prediction_class
            )

        else:

            print()
            print(
                "Original video not found:"
            )

            print(
                original_video_path
            )

        print()


if __name__ == "__main__":
    main()
