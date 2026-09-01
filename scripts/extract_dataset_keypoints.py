import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = Path(
    "manifests/preliminary_53class_manifest.csv"
)

OUTPUT_ROOT = Path("landmarks")

# Number of videos to process.
# None = process the complete manifest.
# For initial testing, use 10.
LIMIT = None


# ============================================================
# LANDMARK DEFINITIONS
# ============================================================

# Exactly 17 Pose landmarks.
#
# We intentionally keep:
#   Nose
#   Eyes
#   Ears
#   Mouth corners
#   Shoulders
#   Elbows
#   Wrists
#
# We do NOT include hips or anything below the hips.

POSE_LANDMARKS = {
    0: "nose",

    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",

    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",

    7: "left_ear",
    8: "right_ear",

    9: "mouth_left",
    10: "mouth_right",

    11: "left_shoulder",
    12: "right_shoulder",

    13: "left_elbow",
    14: "right_elbow",

    15: "left_wrist",
    16: "right_wrist",
}


# All 21 MediaPipe hand landmarks.
HAND_LANDMARKS = list(range(21))


NUM_POSE = len(POSE_LANDMARKS)
NUM_HAND = len(HAND_LANDMARKS)

TOTAL_LANDMARKS = (
    NUM_POSE
    + NUM_HAND
    + NUM_HAND
)

FEATURES_PER_FRAME = TOTAL_LANDMARKS * 3


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmark(landmark):
    """
    Extract x, y, z from one MediaPipe landmark.
    """
    return [
        landmark.x,
        landmark.y,
        landmark.z,
    ]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_landmarks(landmarks):
    """
    Normalize landmarks using the midpoint between the
    left and right shoulders as origin.

    Scale:
        distance between left and right shoulders.

    Missing landmarks remain [0, 0, 0].
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    ).copy()

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2.0

    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder
    )

    if shoulder_distance < 1e-6:
        shoulder_distance = 1.0

    landmarks = (
        landmarks - shoulder_center
    ) / shoulder_distance

    return landmarks


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(row):

    video_path = Path(row["video_path"])

    signer = str(row["signer"])
    class_name = str(row["class"])
    class_dir = class_name.replace("/", "_")
    label = int(row["label"])
    split = str(row["split"])
    view = str(row["view"])
    video_name = Path(str(row["video_name"])).stem

    output_dir = (
        OUTPUT_ROOT
        / signer
        / class_dir
        / video_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    landmarks_path = (
        output_dir / "landmarks.npy"
    )

    landmarks_177_path = (
        output_dir / "landmarks_177.npy"
    )

    metadata_path = (
        output_dir / "metadata.json"
    )

    if (
    landmarks_path.exists()
    and landmarks_177_path.exists()
    and metadata_path.exists()
    ):
        return {
        "status": "skipped",
        "video": video_name
        }


    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    original_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    pose_detected = 0
    left_hand_detected = 0
    right_hand_detected = 0

    all_landmarks = []

    frame_number = 0

    # --------------------------------------------------------
    # MediaPipe Holistic
    # --------------------------------------------------------

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # OpenCV BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(
                rgb_frame
            )

            frame_landmarks = []

            # =================================================
            # POSE — EXACTLY 17 LANDMARKS
            # =================================================

            if results.pose_landmarks:

                pose_detected += 1

                for idx in POSE_LANDMARKS:

                    landmark = (
                        results.pose_landmarks
                        .landmark[idx]
                    )

                    frame_landmarks.append(
                        extract_landmark(landmark)
                    )

            else:

                frame_landmarks.extend(
                    [[0.0, 0.0, 0.0]]
                    * NUM_POSE
                )

            # =================================================
            # LEFT HAND — ALL 21 LANDMARKS
            # =================================================

            if results.left_hand_landmarks:

                left_hand_detected += 1

                for idx in HAND_LANDMARKS:

                    landmark = (
                        results.left_hand_landmarks
                        .landmark[idx]
                    )

                    frame_landmarks.append(
                        extract_landmark(landmark)
                    )

            else:

                frame_landmarks.extend(
                    [[0.0, 0.0, 0.0]]
                    * NUM_HAND
                )

            # =================================================
            # RIGHT HAND — ALL 21 LANDMARKS
            # =================================================

            if results.right_hand_landmarks:

                right_hand_detected += 1

                for idx in HAND_LANDMARKS:

                    landmark = (
                        results.right_hand_landmarks
                        .landmark[idx]
                    )

                    frame_landmarks.append(
                        extract_landmark(landmark)
                    )

            else:

                frame_landmarks.extend(
                    [[0.0, 0.0, 0.0]]
                    * NUM_HAND
                )

            # =================================================
            # VERIFY 59 LANDMARKS
            # =================================================

            assert len(frame_landmarks) == (
                TOTAL_LANDMARKS
            )

            # -------------------------------------------------
            # Normalize
            # -------------------------------------------------

            frame_landmarks = normalize_landmarks(
                frame_landmarks
            )

            all_landmarks.append(
                frame_landmarks
            )

    cap.release()

    # ========================================================
    # Convert to NumPy
    # ========================================================

    landmarks = np.asarray(
        all_landmarks,
        dtype=np.float32
    )

    # Expected:
    #
    # (T, 59, 3)

    assert landmarks.ndim == 3
    assert landmarks.shape[1] == TOTAL_LANDMARKS
    assert landmarks.shape[2] == 3

    # ========================================================
    # Flatten
    # ========================================================

    landmarks_177 = landmarks.reshape(
        landmarks.shape[0],
        FEATURES_PER_FRAME
    )

    assert landmarks_177.shape[1] == (
        FEATURES_PER_FRAME
    )

    # ========================================================
    # SAVE NUMERICAL DATA
    # ========================================================

    np.save(
        landmarks_path,
        landmarks
    )

    np.save(
        landmarks_177_path,
        landmarks_177
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "video_path": str(video_path),
        "video_name": video_name,

        "signer": signer,
        "class": class_name,
        "label": label,
        "split": split,
        "view": view,

        "fps": float(fps),
        "original_frame_count":
            original_frame_count,

        "processed_frame_count":
            int(landmarks.shape[0]),

        "width": width,
        "height": height,

        "pose_landmarks": NUM_POSE,
        "left_hand_landmarks": NUM_HAND,
        "right_hand_landmarks": NUM_HAND,

        "total_landmarks_per_frame":
            TOTAL_LANDMARKS,

        "coordinates_per_landmark": 3,

        "features_per_frame":
            FEATURES_PER_FRAME,

        "pose_detection_frames":
            pose_detected,

        "left_hand_detection_frames":
            left_hand_detected,

        "right_hand_detection_frames":
            right_hand_detected,

        "landmark_shape":
            list(landmarks.shape),

        "flattened_shape":
            list(landmarks_177.shape),

        "normalization": {
            "type":
                "shoulder_center_and_distance",

            "reference":
                "midpoint_of_left_and_right_shoulder",

            "scale":
                "left_right_shoulder_distance"
        }
    }

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # ========================================================
    # RETURN SUMMARY
    # ========================================================

    return {
        "video": video_name,
        "signer": signer,
        "class": class_name,
        "split": split,
        "frames": landmarks.shape[0],
        "shape": tuple(landmarks.shape),
        "pose_detected": pose_detected,
        "left_hand_detected":
            left_hand_detected,
        "right_hand_detected":
            right_hand_detected,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DATASET MEDIAPIPE KEYPOINT EXTRACTION")
    print("=" * 80)

    print(
        f"Manifest : {MANIFEST_PATH}"
    )

    print(
        f"Output   : {OUTPUT_ROOT}"
    )

    print(
        f"Pose landmarks       : {NUM_POSE}"
    )

    print(
        f"Left hand landmarks  : {NUM_HAND}"
    )

    print(
        f"Right hand landmarks : {NUM_HAND}"
    )

    print(
        f"Total landmarks      : {TOTAL_LANDMARKS}"
    )

    print(
        f"Features/frame       : {FEATURES_PER_FRAME}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Read manifest
    # --------------------------------------------------------

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Total videos in manifest: {len(df)}"
    )

    # --------------------------------------------------------
    # Limit for testing
    # --------------------------------------------------------

    if LIMIT is not None:

        df = df.head(LIMIT)

        print(
            f"Testing with first {LIMIT} videos"
        )

    else:

        print(
            "Processing complete dataset"
        )

    print("=" * 80)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    successful = 0
    failed = 0

    failures = []

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        print(
            f"\n[{i}/{len(df)}] "
            f"{row['signer']} | "
            f"{row['class']} | "
            f"{row['video_name']}"
        )

        try:

            result = process_video(row)

            successful += 1

            print(
                f"  Frames : {result['frames']}"
            )

            print(
                f"  Shape  : {result['shape']}"
            )

            print(
                f"  Pose   : "
                f"{result['pose_detected']}/"
                f"{result['frames']}"
            )

            print(
                f"  Left   : "
                f"{result['left_hand_detected']}/"
                f"{result['frames']}"
            )

            print(
                f"  Right  : "
                f"{result['right_hand_detected']}/"
                f"{result['frames']}"
            )

        except Exception as e:

            failed += 1

            failures.append({
                "video":
                    row["video_name"],
                "error":
                    str(e)
            })

            print(
                f"  FAILED: {e}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("DATASET EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"Processed successfully : {successful}"
    )

    print(
        f"Failed                  : {failed}"
    )

    print(
        f"Total attempted         : {len(df)}"
    )

    if failures:

        print("\nFailures:")

        for failure in failures:

            print(
                f"  {failure['video']} "
                f"-> {failure['error']}"
            )

    print("=" * 80)


if __name__ == "__main__":
    main()
