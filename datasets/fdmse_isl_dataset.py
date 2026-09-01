import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FDMSEISLDataset(Dataset):
    """
    FDMSE-ISL landmark dataset.

    Each sample contains:
        landmarks: (T, 177)
        label:     integer in [0, 52]

    The 177 features correspond to:
        59 landmarks × 3 coordinates (X, Y, Z)
    """

    def __init__(self, manifest_path, split):
        self.manifest = pd.read_csv(manifest_path)

        # Keep only the requested signer-independent split.
        self.manifest = self.manifest[
            self.manifest["split"] == split
        ].reset_index(drop=True)

        if len(self.manifest) == 0:
            raise ValueError(
                f"No samples found for split='{split}' "
                f"in {manifest_path}"
            )

        # Basic consistency checks.
        if not (self.manifest["features_per_frame"] == 177).all():
            raise ValueError(
                "Expected all samples to contain 177 features/frame."
            )

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]

        # Load precomputed normalized landmarks.
        landmarks = np.load(row["landmarks_177_path"])

        if landmarks.ndim != 2:
            raise ValueError(
                f"Expected 2D landmark array, got "
                f"shape={landmarks.shape} for {row['video_name']}"
            )

        if landmarks.shape[1] != 177:
            raise ValueError(
                f"Expected 177 features/frame, got "
                f"{landmarks.shape[1]} for {row['video_name']}"
            )

        landmarks = torch.from_numpy(
            landmarks.astype(np.float32)
        )

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return landmarks, label


if __name__ == "__main__":
    manifest = "../manifests/landmark_dataset_index.csv"

    for split in ["train", "val", "test"]:
        dataset = FDMSEISLDataset(manifest, split)

        print(f"\n{split.upper()}")
        print("Samples:", len(dataset))

        x, y = dataset[0]

        print("Sample:", dataset.manifest.iloc[0]["video_name"])
        print("Landmark shape:", tuple(x.shape))
        print("Label:", y.item())
        print("dtype:", x.dtype)
