"""Loading, reshaping, and normalization for the Kaggle Digit Recognizer CSVs."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_labeled_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Kaggle ``train.csv``-style file into images and labels.

    Returns
    -------
    images : np.ndarray, shape (N, 28, 28), float32 in [0, 1]
    labels : np.ndarray, shape (N,), int64
    """
    df = pd.read_csv(csv_path)
    labels = df["label"].to_numpy(dtype=np.int64)
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    images = pixels.reshape(-1, 28, 28) / 255.0
    return images, labels


def load_unlabeled_csv(csv_path: Path) -> np.ndarray:
    """Load a Kaggle ``test.csv``-style file (no label column) into images."""
    df = pd.read_csv(csv_path)
    pixels = df.to_numpy(dtype=np.float32)
    return pixels.reshape(-1, 28, 28) / 255.0


def train_val_split(
    images: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random (non-stratified) train/validation split."""
    rng = np.random.default_rng(seed)
    n_val = int(len(images) * val_fraction)
    perm = rng.permutation(len(images))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


class MNISTDataset(Dataset):
    """Wraps normalized (N, 28, 28) images and (N,) labels as a CHW tensor dataset."""

    def __init__(self, images: np.ndarray, labels: np.ndarray | None = None):
        self.images = torch.from_numpy(images).unsqueeze(1)  # (N, 1, 28, 28)
        self.labels = None if labels is None else torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.images[idx]
        return self.images[idx], self.labels[idx]
