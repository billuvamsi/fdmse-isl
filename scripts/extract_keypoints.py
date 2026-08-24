import cv2
import json
import numpy as np
import mediapipe as mp
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = (
    "/home/billu/FDMSE-ISL/data/"
    "s0004/front/s0004_f_w000043.mp4"
)

SIGNER = "s0004"
CLASS_NAME = "Angry"
VIDEO_NAME = "s0004_f_w000043"

OUTPUT_ROOT = Path("frames")
LANDMARK_ROOT = Path("landmarks")

# ------------------------------------------------------------
# Selected MediaPipe Pose landmarks
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Hand landmarks
# ------------------------------------------------------------

HAND_LANDMARKS = list(range(21))


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

frames_no_keypoints_dir = (
    OUTPUT_ROOT
    / "frames_no_keypoints"
    / SIGNER
    / CLASS_NAME
    / VIDEO_NAME
)

frames_keypoints_dir = (
    OUTPUT_ROOT
    / "frames_keypoints"
    / SIGNER
    / CLASS_NAME
    / VIDEO_NAME
)

landmark_dir = (
    LANDMARK_ROOT
    / SIGNER
    / CLASS_NAME
    / VIDEO_NAME
)

frames_no_keypoints_dir.mkdir(parents=True, exist_ok=True)
frames_keypoints_dir.mkdir(parents=True, exist_ok=True)
landmark_dir.mkdir(parents=True, exist_ok=True)

def normalize_landmarks(landmarks):
    """
    Body-relative normalization.

    Reference:
        Midpoint between left and right shoulders.

    Scale:
        Distance between left and right shoulders.

    Missing landmarks represented by [0, 0, 0]
    remain [0, 0, 0].
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    ).copy()

    # Identify missing landmarks BEFORE normalization
    missing_mask = np.all(
        np.isclose(landmarks, 0.0),
        axis=1
    )

    # Left and right shoulder
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    # Shoulder midpoint
    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2.0

    # Shoulder distance
    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder
    )

    # Avoid division by zero
    if shoulder_distance < 1e-6:
        shoulder_distance = 1.0

    # Normalize
    landmarks = (
        landmarks - shoulder_center
    ) / shoulder_distance

    # Restore missing landmarks
    landmarks[missing_mask] = 0.0

    return landmarks


# ============================================================
# EXTRACT ONE LANDMARK
# ============================================================

def extract_landmark(landmark):
    """
    Extract x, y, z from a MediaPipe landmark.
    """
    return [
        landmark.x,
        landmark.y,
        landmark.z,
    ]


# ============================================================
# DRAW LANDMARKS
# ============================================================

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def draw_selected_landmarks(
    image,
    pose_result,
    left_hand_result,
    right_hand_result,
):
    """
    Draw our selected landmarks on the frame.
    """

    output = image.copy()

    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    if pose_result:

        for idx in POSE_LANDMARKS:

            landmark = pose_result.landmark[idx]

            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])

            # Keep only visible image coordinates
            if (
                0 <= x < image.shape[1]
                and 0 <= y < image.shape[0]
            ):
                cv2.circle(
                    output,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    output,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                )

    # --------------------------------------------------------
    # Left hand
    # --------------------------------------------------------

    if left_hand_result:

        for idx in HAND_LANDMARKS:

            landmark = left_hand_result.landmark[idx]

            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])

            if (
                0 <= x < image.shape[1]
                and 0 <= y < image.shape[0]
            ):
                cv2.circle(
                    output,
                    (x, y),
                    4,
                    (255, 0, 0),
                    -1,
                )

    # --------------------------------------------------------
    # Right hand
    # --------------------------------------------------------

    if right_hand_result:

        for idx in HAND_LANDMARKS:

            landmark = right_hand_result.landmark[idx]

            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])

            if (
                0 <= x < image.shape[1]
                and 0 <= y < image.shape[0]
            ):
                cv2.circle(
                    output,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1,
                )

    return output


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():

    print("=" * 80)
    print("MEDIAPIPE HOLISTIC KEYPOINT EXTRACTION")
    print("=" * 80)

    print("Video :", VIDEO_PATH)
    print("Class :", CLASS_NAME)
    print("Signer:", SIGNER)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print("\nVideo information:")
    print("FPS       :", fps)
    print("Frames    :", total_frames)
    print("Resolution:", f"{width} x {height}")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_landmarks = []

    pose_detected = 0
    left_hand_detected = 0
    right_hand_detected = 0

    frame_number = 0

    # --------------------------------------------------------
    # MediaPipe Holistic
    # --------------------------------------------------------

    with mp.solutions.holistic.Holistic(
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

            # BGR → RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            results = holistic.process(rgb)

            # ------------------------------------------------
            # Extract Pose
            # ------------------------------------------------

            frame_landmarks = []

            if results.pose_landmarks:

                pose_detected += 1

                for idx in POSE_LANDMARKS:

                    landmark = (
                        results.pose_landmarks.landmark[idx]
                    )

                    frame_landmarks.append(
                        extract_landmark(landmark)
                    )

            else:

                frame_landmarks.extend(
                    [[0.0, 0.0, 0.0]]
                    * len(POSE_LANDMARKS)
                )

            # ------------------------------------------------
            # Left hand
            # ------------------------------------------------

            if results.left_hand_landmarks:

                left_hand_detected += 1

                for idx in HAND_LANDMARKS:

                    landmark = (
                        results.left_hand_landmarks.landmark[idx]
                    )

                    frame_landmarks.append(
                        extract_landmark(landmark)
                    )

            else:

                frame_landmarks.extend(
                    [[0.0, 0.0, 0.0]]
                    * 21
                )

            # ------------------------------------------------
            # Right hand
            # ------------------------------------------------

            if results.right_hand_landmarks:

                right_hand_detected += 1

                for idx in HAND_LANDMARKS:

                    landmark = (
                        results.right_hand_landmarks.landmark[idx]
                    )

                    frame_landmarks.append(
                        extract_landmark(landmark)
                    )

            else:

                frame_landmarks.extend(
                    [[0.0, 0.0, 0.0]]
                    * 21
                )

            # ------------------------------------------------
            # Verify 59 landmarks
            # ------------------------------------------------

            assert len(frame_landmarks) == 59

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            frame_landmarks = normalize_landmarks(
                frame_landmarks
            )

            all_landmarks.append(
                frame_landmarks
            )

            # ------------------------------------------------
            # Save original frame
            # ------------------------------------------------

            frame_filename = (
                f"frame_{frame_number:04d}.jpg"
            )

            cv2.imwrite(
                str(
                    frames_no_keypoints_dir
                    / frame_filename
                ),
                frame,
            )

            # ------------------------------------------------
            # Save visualization
            # ------------------------------------------------

            visualization = draw_selected_landmarks(
                frame,
                results.pose_landmarks,
                results.left_hand_landmarks,
                results.right_hand_landmarks,
            )

            cv2.imwrite(
                str(
                    frames_keypoints_dir
                    / frame_filename
                ),
                visualization,
            )

    cap.release()

    # ========================================================
    # SAVE NUMERICAL LANDMARK DATA
    # ========================================================

    all_landmarks = np.asarray(
        all_landmarks,
        dtype=np.float32,
    )

    # Shape:
    # frames × 59 × 3

    npy_path = landmark_dir / "landmarks.npy"

    np.save(
        npy_path,
        all_landmarks,
    )

    # Also save flattened representation:
    # frames × 177

    flattened = all_landmarks.reshape(
        all_landmarks.shape[0],
        59 * 3,
    )

    flattened_path = (
        landmark_dir / "landmarks_177.npy"
    )

    np.save(
        flattened_path,
        flattened,
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "video_path": VIDEO_PATH,
        "signer": SIGNER,
        "class": CLASS_NAME,
        "video_name": VIDEO_NAME,
        "fps": fps,
        "original_frame_count": total_frames,
        "processed_frame_count": frame_number,
        "width": width,
        "height": height,
        "pose_landmarks": 17,
        "left_hand_landmarks": 21,
        "right_hand_landmarks": 21,
        "total_landmarks_per_frame": 59,
        "coordinates_per_landmark": 3,
        "features_per_frame": 177,
        "pose_detection_frames": pose_detected,
        "left_hand_detection_frames": left_hand_detected,
        "right_hand_detection_frames": right_hand_detected,
        "landmark_shape": list(
            all_landmarks.shape
        ),
    }

    metadata_path = (
        landmark_dir / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

    print("Processed frames :", frame_number)

    print(
        "Pose detected    :",
        f"{pose_detected}/{frame_number}",
    )

    print(
        "Left hand detected:",
        f"{left_hand_detected}/{frame_number}",
    )

    print(
        "Right hand detected:",
        f"{right_hand_detected}/{frame_number}",
    )

    print(
        "\nLandmark array shape:",
        all_landmarks.shape,
    )

    print(
        "Flattened shape     :",
        flattened.shape,
    )

    print("\nSaved:")
    print("Original frames:")
    print(frames_no_keypoints_dir)

    print("\nKeypoint frames:")
    print(frames_keypoints_dir)

    print("\nLandmarks:")
    print(npy_path)

    print("\nFlattened 177:")
    print(flattened_path)

    print("\nMetadata:")
    print(metadata_path)

    print("=" * 80)


if __name__ == "__main__":
    main()

