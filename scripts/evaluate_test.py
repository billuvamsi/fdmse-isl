import sys
from pathlib import Path

import numpy as np
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
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    total = 0
    correct = 0

    predictions = []
    targets = []

    for inputs, labels in loader:

        # DataLoader output:
        # (1, T, 177)

        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        predicted = outputs.argmax(dim=1)

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

        predictions.extend(
            predicted.cpu().numpy().tolist()
        )

        targets.extend(
            labels.cpu().numpy().tolist()
        )

    accuracy = 100.0 * correct / total

    return accuracy, predictions, targets


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("FDMSE-ISL SPOTER — FINAL TEST EVALUATION")
    print("=" * 80)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = FDMSEISLDataset(
        MANIFEST,
        split="test"
    )

    print(
        "Test samples:",
        len(test_dataset)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FDMSEISLSPOTER(
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load best validation checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading best checkpoint:"
    )

    print(CHECKPOINT)

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
        "Checkpoint validation accuracy:",
        f"{checkpoint['val_accuracy']:.2f}%"
    )

    print(
        "Checkpoint validation loss:",
        f"{checkpoint['val_loss']:.4f}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\nEvaluating untouched test set...")

    accuracy, predictions, targets = evaluate(
        model,
        test_loader,
        DEVICE
    )

    print()

    print("=" * 80)
    print("TEST RESULT")
    print("=" * 80)

    print(
        f"Test Accuracy: {accuracy:.2f}%"
    )

    print(
        f"Correct: {sum(p == t for p, t in zip(predictions, targets))}"
    )

    print(
        f"Total: {len(targets)}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    for target, prediction in zip(
        targets,
        predictions
    ):
        confusion[target, prediction] += 1

    confusion_path = (
        ROOT
        / "logs"
        / "test_confusion_matrix.npy"
    )

    confusion_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        confusion_path,
        confusion
    )

    print(
        "\nConfusion matrix saved:",
        confusion_path
    )


if __name__ == "__main__":
    main()
