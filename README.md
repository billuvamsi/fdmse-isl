# FDMSE-ISL Recognition

Preliminary implementation repository for isolated Indian Sign Language (ISL) recognition using the FDMSE-ISL dataset and a Transformer-based recognition pipeline (initially targeting SPOTER).

## Current Progress

### Dataset
- Original FDMSE-ISL metadata: `data_meta/metadata.csv`
- Original metadata contains 40,033 video records and 2,002 unique sign classes.
- The current preliminary study uses **front-view** videos because the available metadata provides class labels for the front view only.
- The original dataset is **not included** in this repository.

### Preliminary vocabulary
A manually curated set of **53 daily-use signs** has been selected from the original 2,002-class vocabulary. Numeric signs, symbolic signs, and other non-target classes are excluded from this preliminary experiment.

### Verified preliminary subset
- Classes: 53
- Videos: 1,060
- Videos per class: 20
- Signers: 20
- View: front
- Original metadata split: 530 train / 106 validation / 424 test
- Missing selected classes: 0
- Missing video files: 0 (verified against the original dataset location)

> The original metadata split is retained for reference. A signer-independent split will be defined in the next stage of the experiment.

## Repository Structure

```text
fdmse-isl/
├── README.md
├── .gitignore
├── environment.yml
├── metadata/
│   ├── classes.txt
│   └── selected_daily_classes.txt
├── manifests/
│   └── preliminary_53class_manifest.csv
└── scripts/
    ├── create_classes.py
    ├── create_manifest.py
    └── verify_video_paths.py
```

## Environment

Development environment used on the GPU server:

- Conda environment: `fdmse_isl`
- Python: `3.10.20`

The `environment.yml` file records the Python version and the lightweight packages used so far.

## Planned Pipeline

```text
FDMSE-ISL metadata
        ↓
53 daily-use classes
        ↓
verified front-view manifest
        ↓
signer-independent split
        ↓
MediaPipe landmark extraction
        ↓
54 selected landmarks × XYZ
        ↓
feature preprocessing / normalization
        ↓
SPOTER adaptation
        ↓
training and evaluation
```

## Data Policy

Do **not** commit the original FDMSE-ISL videos, raw dataset files, extracted landmark arrays, model checkpoints, or large generated logs to this repository.
