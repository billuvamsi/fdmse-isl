import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add repository root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.fdmse_isl_dataset import FDMSEISLDataset
from models.fdmse_isl_spoter import FDMSEISLSPOTER


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MANIFEST = ROOT / "manifests" / "landmark_dataset_index.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 53
INPUT_DIM = 177
HIDDEN_DIM = 108

LEARNING_RATE = 1e-4


# ---------------------------------------------------------
# Device information
# ---------------------------------------------------------

print("=" * 70)
print("FDMSE-ISL SPOTER — REAL TRAINING STEP TEST")
print("=" * 70)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU capability:",
        torch.cuda.get_device_capability(0)
    )

print()


# ---------------------------------------------------------
# Load training dataset
# ---------------------------------------------------------

print("Loading training dataset...")

train_dataset = FDMSEISLDataset(
    MANIFEST,
    split="train"
)

print("Training samples:", len(train_dataset))

# Batch size = 1 because sequences have different lengths.
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

print()


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------

print("Creating model...")

model = FDMSEISLSPOTER(
    num_classes=NUM_CLASSES,
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)

print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)

print()


# ---------------------------------------------------------
# Loss and optimizer
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------------------------------------------------
# Get ONE real sample
# ---------------------------------------------------------

print("Loading one real training sample...")

inputs, labels = next(iter(train_loader))

print("Raw input shape:", tuple(inputs.shape))
print("Raw label shape:", tuple(labels.shape))
print("Label:", labels.item())

# DataLoader with batch_size=1 gives:
#
# (1, T, 177)
#
# Remove batch dimension because the model expects:
#
# (T, 177)

inputs = inputs.squeeze(0).to(DEVICE)
labels = labels.to(DEVICE)

print("Model input shape:", tuple(inputs.shape))
print("Model label:", labels.item())

print()


# ---------------------------------------------------------
# Forward pass
# ---------------------------------------------------------

print("Running forward pass...")

optimizer.zero_grad()

outputs = model(inputs)

print("Output shape:", tuple(outputs.shape))

print()


# ---------------------------------------------------------
# Loss
# ---------------------------------------------------------

loss = criterion(
    outputs,
    labels
)

print("Loss:", loss.item())

print()


# ---------------------------------------------------------
# Backward pass
# ---------------------------------------------------------

print("Running backward pass...")

loss.backward()

print("Backward pass successful.")

print()


# ---------------------------------------------------------
# Optimizer update
# ---------------------------------------------------------

print("Updating model parameters...")

optimizer.step()

print("Optimizer step successful.")

print()


# ---------------------------------------------------------
# CUDA synchronization
# ---------------------------------------------------------

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

print("=" * 70)
print("SUCCESS: ONE COMPLETE REAL TRAINING STEP FINISHED")
print("=" * 70)
