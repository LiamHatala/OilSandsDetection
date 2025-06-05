"""
custom_metrics.py

Defines custom evaluation metrics for segmentation and classification tasks:
- Intersection over Union (IoU) for segmentation
- Accuracy for discrete level classification
"""

import torch

def Calculate_IoU(preds, labels):
    """
    Compute the Intersection over Union (IoU) between predicted and true segmentation masks.

    Args:
        preds (Tensor): Binary prediction mask (0 or 1), shape (N, H, W).
        labels (Tensor): Ground truth binary mask, shape (N, H, W).

    Returns:
        Tensor: IoU score (scalar tensor).
    """

    intersection = (preds * labels).sum()
    union = ((preds + labels) > 0).float().sum()
    iou = (intersection + 1e-6) / (union + 1e-6) # Adding epsilon to reduce division by zero
    return iou

def level_accuracy(y_pred_level, y_true_level):
    """
    Compute classification accuracy for predicted vs. ground truth discrete levels.

    Args:
        y_pred_level (Tensor): Predicted levels, shape (N,).
        y_true_level (Tensor): Ground truth levels, shape (N,).

    Returns:
        Tensor: Accuracy (scalar tensor between 0 and 1).
    """

    return (y_pred_level == y_true_level).float().mean()

