import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ============================================================
# Repository setup
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.fdmse_isl_dataset import FDMSEISLDataset
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
# Macro F1
# ============================================================

def calculate_f1_scores(confusion):

    f1_scores = []

    for c in range(NUM_CLASSES):

        tp = confusion[c, c]

        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp

        precision_denominator = tp + fp
        recall_denominator = tp + fn

        if precision_denominator == 0:
            precision = 0.0
        else:
            precision = tp / precision_denominator

        if recall_denominator == 0:
            recall = 0.0
        else:
            recall = tp / recall_denominator

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

        f1_scores.append(f1)

    return np.array(f1_scores)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, dataset, loader, device):

    model.eval()

    predictions = []
    targets = []
    video_names = []
    signers = []
    true_classes = []

    for index, (inputs, labels) in enumerate(loader):

        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        predicted = outputs.argmax(dim=1)

        predictions.append(
            int(predicted.item())
        )

        targets.append(
            int(labels.item())
        )

        row = dataset.manifest.iloc[index]

        video_names.append(
            row["video_name"]
        )

        signers.append(
            row["signer"]
        )

        true_classes.append(
            row["class"]
        )

    return (
        predictions,
        targets,
        video_names,
        signers,
        true_classes,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("DETAILED FDMSE-ISL SPOTER TEST EVALUATION")
    print("=" * 80)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = FDMSEISLDataset(
        MANIFEST,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Test samples:",
        len(test_dataset)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation accuracy:",
        f"{checkpoint['val_accuracy']:.2f}%"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    (
        predictions,
        targets,
        video_names,
        signers,
        true_classes,
    ) = evaluate(
        model,
        test_dataset,
        test_loader,
        DEVICE
    )

    # --------------------------------------------------------
    # Class mapping
    # --------------------------------------------------------

    class_mapping = (
        test_dataset.manifest[
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

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    for true, pred in zip(
        targets,
        predictions
    ):
        confusion[true, pred] += 1

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    correct = sum(
        p == t
        for p, t in zip(
            predictions,
            targets
        )
    )

    accuracy = (
        100.0
        * correct
        / len(targets)
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1_scores = calculate_f1_scores(
        confusion
    )

    macro_f1 = (
        f1_scores.mean()
    )

    # Since the test set is balanced,
    # weighted F1 is very close to macro F1.
    weighted_f1 = (
        np.average(
            f1_scores,
            weights=confusion.sum(axis=1)
        )
    )

    # --------------------------------------------------------
    # Print overall metrics
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("OVERALL TEST METRICS")
    print("=" * 80)

    print(
        f"Accuracy     : {accuracy:.2f}%"
    )

    print(
        f"Macro-F1     : {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1  : {weighted_f1:.4f}"
    )

    print(
        f"Correct      : {correct}/{len(targets)}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    class_rows = []

    for label in range(NUM_CLASSES):

        total_class = confusion[
            label
        ].sum()

        class_correct = confusion[
            label,
            label
        ]

        class_accuracy = (
            100.0
            * class_correct
            / total_class
            if total_class > 0
            else 0.0
        )

        class_rows.append({
            "label": label,
            "class": label_to_class[label],
            "samples": total_class,
            "correct": class_correct,
            "accuracy": class_accuracy,
            "f1": f1_scores[label],
        })

    class_df = pd.DataFrame(
        class_rows
    )

    # --------------------------------------------------------
    # Best classes
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TOP 10 CLASSES")
    print("=" * 80)

    print(
        class_df
        .sort_values(
            ["accuracy", "f1"],
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Worst classes
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BOTTOM 10 CLASSES")
    print("=" * 80)

    print(
        class_df
        .sort_values(
            ["accuracy", "f1"],
            ascending=True
        )
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Strongest confusion pairs
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STRONGEST CONFUSION PAIRS")
    print("=" * 80)

    confusion_pairs = []

    for true in range(NUM_CLASSES):

        for pred in range(NUM_CLASSES):

            if true == pred:
                continue

            count = confusion[
                true,
                pred
            ]

            if count > 0:

                confusion_pairs.append({
                    "true_label": true,
                    "true_class": label_to_class[true],
                    "pred_label": pred,
                    "pred_class": label_to_class[pred],
                    "count": count,
                })

    confusion_df = pd.DataFrame(
        confusion_pairs
    )

    if len(confusion_df) > 0:

        print(
            confusion_df
            .sort_values(
                "count",
                ascending=False
            )
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Save prediction CSV
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({
        "video_name": video_names,
        "signer": signers,
        "true_label": targets,
        "true_class": true_classes,
        "predicted_label": predictions,
        "predicted_class": [
            label_to_class[p]
            for p in predictions
        ],
        "correct": [
            p == t
            for p, t in zip(
                predictions,
                targets
            )
        ],
    })

    predictions_path = (
        ROOT
        / "logs"
        / "test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Save per-class metrics
    # --------------------------------------------------------

    class_metrics_path = (
        ROOT
        / "logs"
        / "test_class_metrics.csv"
    )

    class_df.to_csv(
        class_metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_path = (
        ROOT
        / "logs"
        / "test_confusion_matrix.npy"
    )

    np.save(
        confusion_path,
        confusion
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        "Predictions:",
        predictions_path
    )

    print(
        "Class metrics:",
        class_metrics_path
    )

    print(
        "Confusion matrix:",
        confusion_path
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
