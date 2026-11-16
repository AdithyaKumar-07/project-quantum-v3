"""Optional native-resolution 2D U-Net for four-modality MSD segmentation.

This model is deliberately separate from the RF model used by the FRQI/NEQR
hybrid experiments. It is a stronger spatial-context candidate for classical
segmentation, not a quantum model and not a clinically validated system.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

LABEL_IDS = (0, 1, 2, 4)


def encode_internal_labels(labels: np.ndarray) -> np.ndarray:
    """Map MSD internal IDs {0,1,2,4} to contiguous class indices {0,1,2,3}."""
    arr = np.asarray(labels)
    if not np.all(np.isin(arr, LABEL_IDS)):
        raise ValueError("Labels must use the project's internal IDs {0,1,2,4}")
    lookup = np.zeros(5, dtype=np.uint8)
    lookup[1] = 1
    lookup[2] = 2
    lookup[4] = 3
    return lookup[arr.astype(np.uint8)]


def decode_class_indices(indices: np.ndarray) -> np.ndarray:
    """Map contiguous U-Net outputs back to MSD internal IDs {0,1,2,4}."""
    arr = np.asarray(indices)
    if arr.size and (arr.min() < 0 or arr.max() > 3):
        raise ValueError("U-Net class indices must be in [0,3]")
    return np.asarray(LABEL_IDS, dtype=np.uint8)[arr.astype(np.uint8)]


def select_patient_slice_indices(labels: np.ndarray, max_slices: int = 8,
                                 tumour_fraction: float = 0.75) -> np.ndarray:
    """Select representative positive and negative slices per patient.

    Rare subregions get a representative slice first. Remaining slots are
    distributed across tumour-positive slices and then background slices.
    """
    y = np.asarray(labels)
    if y.ndim != 3 or max_slices < 1 or not (0.0 < tumour_fraction <= 1.0):
        raise ValueError("Expected labels (Z,H,W), positive max_slices and tumour_fraction in (0,1]")
    positive = np.flatnonzero(np.any(y > 0, axis=(1, 2)))
    negative = np.flatnonzero(~np.any(y > 0, axis=(1, 2)))
    if not len(positive):
        pick = np.linspace(0, len(y) - 1, min(max_slices, len(y))).round().astype(int)
        return np.unique(pick)
    mandatory = []
    for cls in (1, 2, 4):
        counts = np.count_nonzero(y == cls, axis=(1, 2))
        if np.any(counts):
            mandatory.append(int(np.argmax(counts)))
    chosen = list(dict.fromkeys(mandatory))[:max_slices]
    positive_target = min(len(positive), max(len(chosen), int(round(max_slices * tumour_fraction))))
    remaining_pos = [int(z) for z in positive if int(z) not in chosen]
    slots = max(0, positive_target - len(chosen))
    if slots and remaining_pos:
        ids = np.linspace(0, len(remaining_pos) - 1, min(slots, len(remaining_pos))).round().astype(int)
        chosen.extend(remaining_pos[i] for i in np.unique(ids))
    slots = max(0, max_slices - len(chosen))
    if slots and len(negative):
        ids = np.linspace(0, len(negative) - 1, min(slots, len(negative))).round().astype(int)
        chosen.extend(int(negative[i]) for i in np.unique(ids))
    slots = max(0, min(max_slices, len(y)) - len(chosen))
    if slots and len(positive):
        remaining = [int(z) for z in positive if int(z) not in chosen]
        if remaining:
            ids = np.linspace(0, len(remaining) - 1, min(slots, len(remaining))).round().astype(int)
            chosen.extend(remaining[i] for i in np.unique(ids))
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # Keep the RF/API project importable without the optional DL extra.
    torch = None
    nn = None
    F = None


if nn is not None:
    def _group_count(channels: int) -> int:
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1


    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.08):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
            )
            self.skip = (nn.Identity() if in_channels == out_channels else
                         nn.Conv2d(in_channels, out_channels, 1, bias=False))

        def forward(self, x):
            return self.block(x) + self.skip(x)


    class UNet2D(nn.Module):
        """Compact residual U-Net. Input: (B,4,H,W); output: (B,4,H,W)."""
        def __init__(self, in_channels: int = 4, num_classes: int = 4,
                     base_channels: int = 16, dropout: float = 0.08):
            super().__init__()
            if in_channels != 4 or num_classes != 4 or base_channels < 8:
                raise ValueError("This project U-Net requires 4 input/output channels and base_channels >= 8")
            c = base_channels
            self.enc1 = ConvBlock(in_channels, c, dropout)
            self.enc2 = ConvBlock(c, c * 2, dropout)
            self.enc3 = ConvBlock(c * 2, c * 4, dropout)
            self.enc4 = ConvBlock(c * 4, c * 8, dropout)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(c * 8, c * 16, dropout)
            self.dec4 = ConvBlock(c * 16 + c * 8, c * 8, dropout)
            self.dec3 = ConvBlock(c * 8 + c * 4, c * 4, dropout)
            self.dec2 = ConvBlock(c * 4 + c * 2, c * 2, dropout)
            self.dec1 = ConvBlock(c * 2 + c, c, dropout)
            self.head = nn.Conv2d(c, num_classes, kernel_size=1)

        @staticmethod
        def _up(x, reference):
            return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

        def forward(self, x):
            if x.ndim != 4 or x.shape[1] != 4:
                raise ValueError("U-Net input must have shape (B,4,H,W)")
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self._up(b, e4), e4], dim=1))
            d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
            d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
            d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
            return self.head(d1)


    def region_dice_loss(logits, targets):
        """Dice loss over the overlapping WT, TC and ET regions."""
        probs = torch.softmax(logits, dim=1)
        truth = F.one_hot(targets.long(), num_classes=4).permute(0, 3, 1, 2).float()
        pred_regions = torch.stack((probs[:, 1:].sum(dim=1),
                                    probs[:, 1] + probs[:, 3], probs[:, 3]), dim=1)
        true_regions = torch.stack((truth[:, 1:].sum(dim=1),
                                    truth[:, 1] + truth[:, 3], truth[:, 3]), dim=1)
        dims = (0, 2, 3)
        intersection = (pred_regions * true_regions).sum(dims)
        denominator = pred_regions.sum(dims) + true_regions.sum(dims)
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return 1.0 - dice.mean()


    def combined_segmentation_loss(logits, targets, class_weights=None):
        weights = class_weights
        ce = F.cross_entropy(logits, targets.long(), weight=weights)
        dice = region_dice_loss(logits, targets)
        return 0.65 * ce + 0.35 * dice
else:
    class UNet2D:  # pragma: no cover - only used to produce a clear optional-dependency message
        def __init__(self, *args, **kwargs):
            raise ImportError("The optional U-Net requires PyTorch. Install requirements-unet.txt first.")


def _require_torch():
    if torch is None or nn is None:
        raise ImportError("The optional U-Net requires PyTorch. Install requirements-unet.txt first.")


def choose_device(requested: str = "auto") -> str:
    _require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    return requested


@lru_cache(maxsize=4)
def _load_cached_model(model_path: str, mtime_ns: int, device_name: str):
    _require_torch()
    device = torch.device(device_name)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = checkpoint.get("model_config", {})
    model = UNet2D(base_channels=int(config.get("base_channels", 16)),
                   dropout=float(config.get("dropout", 0.08)))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("metadata", {})


def load_unet_model(model_path: str | Path, device: str = "auto"):
    _require_torch()
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"U-Net checkpoint not found: {path}")
    selected_device = choose_device(device)
    return _load_cached_model(str(path), path.stat().st_mtime_ns, selected_device)


def predict_unet_slice(modalities: np.ndarray, model_path: str | Path, *, device: str = "auto") -> np.ndarray:
    """Predict a native-resolution slice; return labels in {0,1,2,4}."""
    _require_torch()
    x = np.asarray(modalities, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] != 4 or x.shape[1] % 16 or x.shape[2] % 16:
        raise ValueError("Expected (4,H,W) modalities with H and W divisible by 16")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input MRI slice contains non-finite values")
    model, _ = load_unet_model(model_path, device)
    device_name = next(model.parameters()).device
    with torch.inference_mode():
        tensor = torch.from_numpy(x[None]).to(device_name)
        logits = model(tensor)
        indices = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return decode_class_indices(indices)


def clear_unet_model_cache() -> None:
    _load_cached_model.cache_clear()
