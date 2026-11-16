"""Validation-only probability calibration for pixel-wise tumour segmentation.

The training sampler intentionally balances classes. This can make raw argmax
inference over-predict tumour pixels at the real, highly imbalanced prevalence.
This module learns a single tumour-vs-background probability threshold on the
validation patients only; it never inspects held-out test labels while selecting
that threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

from .features import extract_features

REGION_LABELS = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}


def probabilities_to_labels(probabilities: np.ndarray, classes: np.ndarray,
                             threshold: float) -> np.ndarray:
    """Convert multiclass probabilities into labels using a tumour confidence gate."""
    probs = np.asarray(probabilities, dtype=np.float32)
    class_values = np.asarray(classes, dtype=np.uint8)
    if probs.ndim != 2 or probs.shape[1] != len(class_values):
        raise ValueError("Probability matrix and class vector are incompatible")
    if not np.isfinite(probs).all():
        raise ValueError("Model probabilities contain non-finite values")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Tumour threshold must lie in [0,1]")
    bg = np.flatnonzero(class_values == 0)
    tumour = np.flatnonzero(class_values != 0)
    if bg.size != 1 or tumour.size == 0:
        raise ValueError("Model must contain background class 0 and at least one tumour class")
    tumour_probability = probs[:, tumour].sum(axis=1)
    winner_idx = tumour[np.argmax(probs[:, tumour], axis=1)]
    labels = class_values[winner_idx].copy()
    labels[tumour_probability < threshold] = 0
    return labels


def _dice(pred: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    tp = int(np.count_nonzero(p & t))
    denom = int(np.count_nonzero(p)) + int(np.count_nonzero(t))
    return 1.0 if denom == 0 else (2.0 * tp / denom)


def slice_macro_dice(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Three-region macro Dice; both-empty regions follow the project's Dice=1 rule."""
    pred = np.asarray(pred_labels)
    truth = np.asarray(true_labels)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and truth shapes differ")
    values = []
    for region, labels in REGION_LABELS.items():
        values.append(_dice(np.isin(pred, labels), np.isin(truth, labels)))
    return float(np.mean(values))


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    validation_macro_dice: float
    baseline_validation_macro_dice: float
    validation_slices: int
    candidates: tuple[dict, ...]


def calibrate_tumour_threshold(model, validation_slices: Iterable[tuple[str, np.ndarray, np.ndarray]],
                                thresholds: Iterable[float] | None = None) -> CalibrationResult:
    """Choose a single tumour probability threshold using validation slices only.

    Each item is (sample_id, modalities=(4,H,W), labels=(H,W)). The objective is
    the mean per-slice WT/TC/ET macro Dice. Ties within 1e-6 prefer the higher
    threshold to avoid unnecessary false-positive tumour area.
    """
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("Calibration requires a classifier exposing predict_proba and classes_")
    grid = np.asarray(list(thresholds) if thresholds is not None else
                      [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                       0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
                       0.925, 0.95, 0.975, 0.99], dtype=float)
    if grid.ndim != 1 or not len(grid) or np.any((grid < 0) | (grid > 1)):
        raise ValueError("Threshold grid must contain values in [0,1]")
    sums = np.zeros(len(grid), dtype=np.float64)
    predicted_coverage = np.zeros(len(grid), dtype=np.float64)
    baseline_sum = 0.0
    count = 0
    for _sample_id, modalities, truth in validation_slices:
        features = extract_features(modalities)
        probabilities = model.predict_proba(features)
        classes = np.asarray(model.classes_, dtype=np.uint8)
        if probabilities.shape[0] != np.asarray(truth).size:
            raise ValueError("Validation probabilities do not align with the label map")
        # Compute per-pixel tumour confidence and the most likely tumour subtype once.
        tumour_columns = np.flatnonzero(classes != 0)
        if not np.any(classes == 0) or tumour_columns.size == 0:
            raise ValueError("Model classes must include background and tumour labels")
        confidence = probabilities[:, tumour_columns].sum(axis=1)
        winner_columns = tumour_columns[np.argmax(probabilities[:, tumour_columns], axis=1)]
        winners = classes[winner_columns]
        baseline_labels = classes[np.argmax(probabilities, axis=1)].reshape(np.asarray(truth).shape)
        truth = np.asarray(truth, dtype=np.uint8)
        baseline_sum += slice_macro_dice(baseline_labels, truth)
        for idx, threshold in enumerate(grid):
            pred = winners.copy()
            pred[confidence < threshold] = 0
            pred = pred.reshape(truth.shape)
            sums[idx] += slice_macro_dice(pred, truth)
            predicted_coverage[idx] += float(np.mean(pred != 0))
        count += 1
    if count == 0:
        raise ValueError("No validation slices were available for calibration")
    means = sums / count
    coverage = predicted_coverage / count
    best_score = float(np.max(means))
    # Within numerical ties, choose the more conservative (higher) threshold.
    winners = np.flatnonzero(means >= best_score - 1e-6)
    best_idx = int(winners[-1])
    details = tuple({"threshold": float(t), "mean_macro_dice": float(s),
                     "mean_predicted_tumour_fraction": float(c)}
                    for t, s, c in zip(grid, means, coverage))
    return CalibrationResult(float(grid[best_idx]), float(means[best_idx]),
                             float(baseline_sum / count), count, details)
