import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# Repository setup
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


from datasets.fdmse_isl_dataset import FDMSEISLDataset
from models.fdmse_isl_spoter import FDMSEISLSPOTER


# ============================================================
# Configuration
# ============================================================

MANIFEST = ROOT / "manifests" / "landmark_dataset_index.csv"

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_fdmse_isl_spoter.pth"


# Model
NUM_CLASSES = 53
INPUT_DIM = 177
HIDDEN_DIM = 108


# Training
BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Early stopping
PATIENCE = 10


# Reproducibility
SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Utility: run one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:

        # ----------------------------------------------------
        # Batch size = 1
        #
        # DataLoader:
        # (1, T, 177)
        #
        # Model:
        # (T, 177)
        # ----------------------------------------------------

        inputs = inputs.squeeze(0).to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            dtype=torch.long,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(inputs)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = total_loss / len(loader)

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:

        inputs = inputs.squeeze(0).to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            dtype=torch.long,
            non_blocking=True
        )

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = total_loss / len(loader)

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("FDMSE-ISL SPOTER TRAINING")
    print("=" * 80)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

        print(
            "GPU capability:",
            torch.cuda.get_device_capability(0)
        )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("Loading datasets...")

    train_dataset = FDMSEISLDataset(
        MANIFEST,
        split="train"
    )

    val_dataset = FDMSEISLDataset(
        MANIFEST,
        split="val"
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    print()

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Creating model...")

    model = FDMSEISLSPOTER(
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        trainable_parameters
    )

    print()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_val_loss = float("inf")

    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
        })

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:6.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        #
        # Primary criterion:
        # validation accuracy
        #
        # Tie-breaker:
        # validation loss
        # ----------------------------------------------------

        improved = (
            val_accuracy > best_val_accuracy
            or (
                val_accuracy == best_val_accuracy
                and val_loss < best_val_loss
            )
        )

        if improved:

            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "train_accuracy": train_accuracy,
                    "train_loss": train_loss,
                    "config": {
                        "num_classes": NUM_CLASSES,
                        "input_dim": INPUT_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "seed": SEED,
                    },
                    "history": history,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"  -> Best model saved "
                f"(Val Acc: {val_accuracy:.2f}%)"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  -> No improvement "
                f"({epochs_without_improvement}/{PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print()
            print(
                f"Early stopping triggered after "
                f"{PATIENCE} epochs without improvement."
            )

            break

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy:.2f}%"
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "Best model:",
        BEST_MODEL_PATH
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
