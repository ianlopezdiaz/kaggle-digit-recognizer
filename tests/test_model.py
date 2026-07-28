import numpy as np
import torch

from digitrecognizer.evaluate import compute_confusion_matrix, misclassified_indices
from digitrecognizer.model import SmallCNN


def test_smallcnn_output_shape():
    model = SmallCNN(num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_compute_confusion_matrix_shape():
    labels = np.array([0, 1, 2, 1])
    preds = np.array([0, 1, 1, 1])
    cm = compute_confusion_matrix(labels, preds)
    assert cm.shape == (10, 10)
    assert cm.sum() == len(labels)


def test_misclassified_indices():
    labels = np.array([0, 1, 2, 3])
    preds = np.array([0, 1, 1, 3])
    assert list(misclassified_indices(labels, preds)) == [2]
