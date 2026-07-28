"""Evaluation beyond accuracy: confusion matrix and misclassified examples."""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return predicted class labels for every example in ``loader``, in order."""
    model.eval()
    preds = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = model(images.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def compute_confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """10x10 confusion matrix, rows are true labels, columns are predictions."""
    return confusion_matrix(labels, preds, labels=list(range(10)))


def misclassified_indices(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Indices where the prediction disagrees with the true label."""
    return np.flatnonzero(labels != preds)


def correct_indices(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Indices where the prediction agrees with the true label."""
    return np.flatnonzero(labels == preds)
