"""Pixel-level multi-modal features for the supervised 2D random-forest model."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

FEATURE_NAMES = (
    "flair", "t1", "t1ce", "t2",
    "flair_smooth", "t1_smooth", "t1ce_smooth", "t2_smooth",
    "flair_gradient", "t1ce_gradient", "x_coordinate", "y_coordinate",
)


def extract_features(modalities: np.ndarray, *, coordinate_origin: tuple[int, int] = (0, 0),
                     coordinate_full_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return (H*W, 12) features, optionally using coordinates in a full image.

    ``coordinate_origin`` is (row, column) of this crop in the full image.
    ``coordinate_full_shape`` is (full_height, full_width). Defaults preserve
    the original local-coordinate behavior used during model training.
    """
    images = np.asarray(modalities, dtype=np.float32)
    if images.ndim != 3 or images.shape[0] != 4:
        raise ValueError("Expected modalities with shape (4,H,W)")
    if not np.all(np.isfinite(images)):
        raise ValueError("Input modalities contain NaN or infinite values")
    _, h, w = images.shape
    smooth = [gaussian_filter(im, sigma=1.0).astype(np.float32) for im in images]
    grad_flair = np.hypot(sobel(images[0], axis=0), sobel(images[0], axis=1))
    grad_t1ce = np.hypot(sobel(images[2], axis=0), sobel(images[2], axis=1))
    yy, xx = np.mgrid[0:h, 0:w]
    row0, col0 = map(int, coordinate_origin)
    full_h, full_w = coordinate_full_shape or (h, w)
    if full_h < 1 or full_w < 1 or row0 < 0 or col0 < 0:
        raise ValueError("coordinate_full_shape and coordinate_origin must be non-negative and non-empty")
    if row0 + h > full_h or col0 + w > full_w:
        raise ValueError("Feature crop lies outside coordinate_full_shape")
    xcoord = ((xx + col0) / max(full_w - 1, 1)).astype(np.float32)
    ycoord = ((yy + row0) / max(full_h - 1, 1)).astype(np.float32)
    stack = np.stack([*images, *smooth, grad_flair, grad_t1ce, xcoord, ycoord], axis=-1)
    return stack.reshape(-1, stack.shape[-1]).astype(np.float32)
