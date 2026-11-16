"""Supervised multi-class pixel classifier and patient-level training helpers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
from .features import extract_features, FEATURE_NAMES

CLASSES = np.array([0, 1, 2, 4], dtype=np.uint8)

@dataclass
class TrainingReport:
    n_samples: int
    class_counts: dict[int, int]
    n_features: int


def balanced_pixel_samples(slices: Iterable[tuple[np.ndarray, np.ndarray]],
                           samples_per_class: int = 500, random_state: int = 42,
                           include_absent_background: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Build a class-balanced pixel sample matrix from aligned 2D slices."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    rng = check_random_state(random_state)
    x_parts, y_parts = [], []
    for modalities, labels in slices:
        labels = np.asarray(labels, dtype=np.uint8)
        if labels.shape != np.asarray(modalities).shape[1:]:
            raise ValueError("Label and modality dimensions do not match")
        features = extract_features(modalities)
        flat_y = labels.ravel()
        for cls in CLASSES:
            indices = np.flatnonzero(flat_y == cls)
            if len(indices) == 0:
                continue
            take = min(samples_per_class, len(indices))
            if cls == 0 and not include_absent_background:
                # Background still participates; this flag is retained for reproducibility.
                continue
            selected = rng.choice(indices, size=take, replace=False)
            x_parts.append(features[selected])
            y_parts.append(flat_y[selected])
    if not x_parts:
        raise ValueError("No labelled pixels found in supplied slices")
    X = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.uint8)
    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain at least two label classes")
    return X, y


def train_random_forest(X: np.ndarray, y: np.ndarray, *, n_estimators: int = 120,
                        max_depth: int | None = 18, random_state: int = 42,
                        n_jobs: int = -1) -> tuple[RandomForestClassifier, TrainingReport]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.uint8)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2D and y must be 1D with matching rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("Training features contain non-finite values")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    clf.fit(X, y)
    unique, counts = np.unique(y, return_counts=True)
    report = TrainingReport(len(y), {int(k): int(v) for k, v in zip(unique, counts)}, X.shape[1])
    return clf, report


def predict_slice(model, modalities: np.ndarray, *, coordinate_origin: tuple[int, int] = (0, 0),
                  coordinate_full_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Predict one slice, respecting validation-fitted calibration when present.

    Optional coordinate arguments preserve global spatial features when callers
    classify image crops/tiles. Existing full-slice callers remain unchanged.
    """
    features = extract_features(modalities, coordinate_origin=coordinate_origin,
                                coordinate_full_shape=coordinate_full_shape)
    calibration = getattr(model, "_qmi_calibration", None)
    if (calibration and calibration.get("accepted", True)
            and calibration.get("tumour_probability_threshold") is not None
            and hasattr(model, "predict_proba")):
        from .calibration import probabilities_to_labels
        probabilities = model.predict_proba(features)
        pred = probabilities_to_labels(
            probabilities, np.asarray(model.classes_),
            float(calibration["tumour_probability_threshold"]),
        )
    else:
        pred = model.predict(features).astype(np.uint8)
    return pred.reshape(np.asarray(modalities).shape[1:])


def save_model(model, path: str | Path, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "feature_names": list(FEATURE_NAMES), "metadata": metadata or {}}
    joblib.dump(payload, path, compress=3)
    return path


def load_model(path: str | Path):
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Invalid model artifact")
    if payload.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Model feature schema does not match this project version")
    return payload["model"], payload.get("metadata", {})
