"""Deterministic synthetic MRI-like slices for smoke tests and UI demonstration.

These samples are NOT medical data and MUST NOT be used as research evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

MODALITIES = ("flair", "t1", "t1ce", "t2")
BRATS_LABELS = (0, 1, 2, 4)

@dataclass
class SliceSample:
    modalities: np.ndarray  # (4, H, W), order FLAIR, T1, T1ce, T2
    label: np.ndarray       # (H, W), values 0, 1, 2, 4
    sample_id: str = "synthetic"


def make_synthetic_slice(seed: int = 0, size: int = 64) -> SliceSample:
    """Create a reproducible, toy multi-modal brain/tumour slice."""
    if size < 32:
        raise ValueError("size must be at least 32")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    cx, cy = rng.uniform(-0.08, 0.08, size=2)
    rx, ry = rng.uniform(0.66, 0.78), rng.uniform(0.70, 0.82)
    brain = (((xx-cx)/rx)**2 + ((yy-cy)/ry)**2) <= 1.0
    # Slightly vary the lesion location while keeping it inside the brain ellipse.
    tx, ty = rng.uniform(-0.18, 0.18), rng.uniform(-0.16, 0.16)
    angle = rng.uniform(-0.6, 0.6)
    xr = (xx-tx)*np.cos(angle) + (yy-ty)*np.sin(angle)
    yr = -(xx-tx)*np.sin(angle) + (yy-ty)*np.cos(angle)
    outer = ((xr/0.30)**2 + (yr/0.23)**2) <= 1.0
    core = ((xr/0.19)**2 + (yr/0.145)**2) <= 1.0
    necrotic = ((xr/0.085)**2 + (yr/0.07)**2) <= 1.0
    enhancing = (((xr-0.035)/0.075)**2 + ((yr+0.015)/0.055)**2) <= 1.0
    # BraTS-style label semantics: 2=edema, 1=necrotic/non-enhancing core, 4=ET.
    labels = np.zeros((size, size), dtype=np.uint8)
    labels[brain & outer] = 2
    labels[brain & core & ~necrotic] = 1
    labels[brain & necrotic] = 1
    labels[brain & enhancing] = 4
    # Background and healthy tissue have different modality signatures.
    tissue = brain.astype(np.float32) * (0.38 + 0.06*np.exp(-((xx/0.55)**2+(yy/0.6)**2)))
    flair = tissue.copy()
    t1 = tissue.copy() * 0.82 + 0.05
    t1ce = tissue.copy() * 0.78 + 0.06
    t2 = tissue.copy() * 0.95 + 0.04
    # Region-dependent contrast, designed only for pipeline smoke tests.
    edema = labels == 2
    core_mask = labels == 1
    et = labels == 4
    flair[edema] += 0.30
    flair[core_mask] += 0.20
    flair[et] += 0.25
    t1[edema] -= 0.08
    t1[core_mask] -= 0.12
    t1[et] += 0.16
    t1ce[edema] += 0.04
    t1ce[core_mask] += 0.10
    t1ce[et] += 0.42
    t2[edema] += 0.34
    t2[core_mask] += 0.20
    t2[et] += 0.16
    images = []
    for image in (flair, t1, t1ce, t2):
        noise = rng.normal(0.0, 0.025, size=(size, size)).astype(np.float32)
        blurred = gaussian_filter(noise, sigma=0.6)
        arr = np.clip(image + noise + blurred, 0.0, 1.0)
        arr[~brain] = 0.0
        images.append(arr.astype(np.float32))
    return SliceSample(np.stack(images, axis=0), labels, f"synthetic_{seed:04d}")


def make_synthetic_dataset(n_samples: int = 48, size: int = 64, seed: int = 42) -> list[SliceSample]:
    if n_samples < 8:
        raise ValueError("n_samples must be at least 8 for train/validation/test splits")
    return [make_synthetic_slice(seed + i, size) for i in range(n_samples)]
