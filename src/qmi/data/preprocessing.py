"""Aligned 2D slice preprocessing for quantum and native-resolution experiments."""
from __future__ import annotations
import numpy as np
import cv2

PROJECT_RESOLUTIONS = (16, 32, 240)


def _occupancy_preserving_labels(labels: np.ndarray, size: int) -> np.ndarray:
    """Downsample labels without silently deleting tiny tumour classes.

    Each output cell receives the most represented tumour class among source
    pixels mapped into it; background is assigned only when no tumour class is
    present. This is a deliberate lesion-preserving low-resolution convention,
    not nearest-neighbour ground truth. Report the convention in publications.
    """
    source = np.asarray(labels, dtype=np.uint8)
    scores = {
        cls: cv2.resize((source == cls).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        for cls in (1, 2, 4)
    }
    result = np.zeros((size, size), dtype=np.uint8)
    best = np.zeros((size, size), dtype=np.float32)
    # Stable tie-break: enhancing > non-enhancing > edema.
    for cls in (2, 1, 4):
        score = scores[cls]
        chosen = score > best
        result[chosen] = cls
        best[chosen] = score[chosen]
    return result


def resize_slice(modalities: np.ndarray, labels: np.ndarray, size: int, *, label_mode: str = "occupancy") -> tuple[np.ndarray, np.ndarray]:
    if modalities.ndim != 3 or modalities.shape[0] != 4:
        raise ValueError("modalities must have shape (4,H,W): FLAIR,T1,T1ce,T2")
    labels = np.asarray(labels)
    if labels.shape != modalities.shape[1:]:
        raise ValueError("label map must match modality spatial dimensions")
    if size not in PROJECT_RESOLUTIONS:
        raise ValueError(f"Resolution must be one of {PROJECT_RESOLUTIONS}")
    if label_mode not in {"occupancy", "nearest"}:
        raise ValueError("label_mode must be 'occupancy' or 'nearest'")
    if not np.all(np.isfinite(modalities)) or not np.all(np.isin(labels, [0, 1, 2, 4])):
        raise ValueError("Modalities must be finite and labels must use {0,1,2,4}")
    if size == 240:
        if modalities.shape[1:] != (240, 240):
            raise ValueError(f"Native mode expects 240x240 slices, got {modalities.shape[1:]}")
        # Copy only the selected native slice so it does not retain the full 4D patient volume.
        return np.clip(modalities.astype(np.float32, copy=False), 0, 1).copy(), labels.astype(np.uint8, copy=False).copy()
    if modalities.shape[1:] == (size, size):
        # Copy only the selected native slice so it does not retain the full 4D patient volume.
        return np.clip(modalities.astype(np.float32, copy=False), 0, 1).copy(), labels.astype(np.uint8, copy=False).copy()
    resized = np.stack([
        cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA if size < m.shape[0] else cv2.INTER_LINEAR).astype(np.float32)
        for m in modalities
    ])
    if size < labels.shape[0] and label_mode == "occupancy":
        mask = _occupancy_preserving_labels(labels, size)
    else:
        mask = cv2.resize(labels.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return np.clip(resized, 0, 1), mask.astype(np.uint8)


def brain_mask(modalities: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Approximate skull-stripped foreground from any nonzero modality."""
    return np.any(np.asarray(modalities) > eps, axis=0)
