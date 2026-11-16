"""Native reader for Medical Segmentation Decathlon Task01_BrainTumour.

Source files are never rewritten. Images are 4-D NIfTI arrays (X,Y,Z,4),
labels are 3-D NIfTI arrays (X,Y,Z). The reader returns modalities in
FLAIR,T1,T1ce,T2 order and labels in the project's internal convention:
0 background, 1 non-enhancing tumour, 2 edema, 4 enhancing tumour.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .msd import modality_index_map, label_index_map, remap_msd_labels


@dataclass(frozen=True)
class MSDCase:
    case_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class MSDMetadata:
    root: Path
    modalities: dict[str, int]
    labels: dict[int, int]
    raw: dict[str, Any]


def _nifti_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in (*folder.glob("*.nii.gz"), *folder.glob("*.nii"))
        if not p.name.startswith("._")
    )


def read_msd_metadata(root: str | Path) -> MSDMetadata:
    base = Path(root).expanduser().resolve()
    metadata_path = base / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"MSD dataset.json not found: {metadata_path}")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw.get("modality"), dict) or not isinstance(raw.get("labels"), dict):
        raise ValueError("dataset.json must contain 'modality' and 'labels' dictionaries")
    return MSDMetadata(base, modality_index_map(raw["modality"]), label_index_map(raw["labels"]), raw)


def discover_msd_cases(root: str | Path, *, require_labels: bool = True) -> list[MSDCase]:
    """Discover labelled train cases without loading the (large) NIfTI volumes."""
    base = Path(root).expanduser().resolve()
    images_dir = base / "imagesTr"
    labels_dir = base / "labelsTr"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"MSD imagesTr directory not found: {images_dir}")
    image_files = _nifti_files(images_dir)
    if not image_files:
        raise FileNotFoundError(f"No .nii or .nii.gz image files found under {images_dir}")
    cases: list[MSDCase] = []
    missing: list[str] = []
    for image_path in image_files:
        label_path = labels_dir / image_path.name
        if require_labels and not label_path.is_file():
            missing.append(image_path.name)
            continue
        case_id = image_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        cases.append(MSDCase(case_id, image_path, label_path if label_path.exists() else Path()))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images have no matching label in labelsTr (first: {missing[0]}). "
            "Training/evaluation require labelled imagesTr/labelsTr pairs."
        )
    if require_labels and not labels_dir.is_dir():
        raise FileNotFoundError(f"MSD labelsTr directory not found: {labels_dir}")
    return cases


def _robust_normalize(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    valid = np.isfinite(arr) & (arr != 0)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    low, high = np.percentile(arr[valid], [1.0, 99.0])
    if not np.isfinite(low + high) or high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - low) / (high - low), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def load_msd_case(case: MSDCase, metadata: MSDMetadata | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load one case; return X=(4,Z,H,W), y=(Z,H,W), with aligned axial slices."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("NIfTI loading requires nibabel; install the project's requirements.txt") from exc
    if metadata is None:
        # Dataset metadata is located two levels above imagesTr/case.nii.gz.
        root = case.image_path.parent.parent
        metadata = read_msd_metadata(root)
    image = nib.as_closest_canonical(nib.load(str(case.image_path)))
    label = nib.as_closest_canonical(nib.load(str(case.label_path)))
    if len(image.shape) != 4 or image.shape[3] != 4:
        raise ValueError(f"{case.case_id}: expected image shape (X,Y,Z,4), got {image.shape}")
    if len(label.shape) != 3 or tuple(image.shape[:3]) != tuple(label.shape):
        raise ValueError(f"{case.case_id}: image/label spatial shapes do not match: {image.shape} vs {label.shape}")
    if not np.allclose(np.asarray(image.affine), np.asarray(label.affine), rtol=0, atol=1e-3):
        raise ValueError(f"{case.case_id}: image and label NIfTI affines are not aligned")
    raw_image = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    raw_label = np.asarray(label.get_fdata(dtype=np.float32), dtype=np.float32)
    # Validate raw labels before conversion; source files stay unchanged.
    seg = remap_msd_labels(raw_label, metadata.labels)
    ordered = []
    for modality_name in ("flair", "t1", "t1ce", "t2"):
        channel = raw_image[..., metadata.modalities[modality_name]]
        if not np.all(np.isfinite(channel)):
            raise ValueError(f"{case.case_id}: {modality_name} contains non-finite voxels")
        ordered.append(_robust_normalize(channel))
    # (X,Y,Z,4) -> (4,Z,X,Y), keeping one axial slice on axis 1.
    modalities = np.stack(ordered, axis=0)
    modalities = np.moveaxis(modalities, -1, 1)
    labels = np.moveaxis(seg, -1, 0)
    if modalities.shape[1:] != labels.shape:
        raise RuntimeError("Internal alignment error after converting MSD volume to axial slices")
    return modalities.astype(np.float32), labels.astype(np.uint8)


def case_to_slices(modalities: np.ndarray, labels: np.ndarray, *, max_slices: int | None = 8,
                   tumor_first: bool = True, return_indices: bool = False):
    """Select representative axial slices; all selection is deterministic."""
    m = np.asarray(modalities)
    y = np.asarray(labels)
    if m.ndim != 4 or m.shape[0] != 4 or y.ndim != 3 or m.shape[1:] != y.shape:
        raise ValueError("Expected modalities (4,Z,H,W) and aligned labels (Z,H,W)")
    indices = np.arange(y.shape[0], dtype=int)
    if tumor_first:
        positive = indices[np.any(y > 0, axis=(1, 2))]
        negative = indices[~np.isin(indices, positive)]
        representatives = []
        for cls in (1, 2, 4):
            counts = np.count_nonzero(y == cls, axis=(1, 2))
            if np.any(counts):
                representatives.append(int(np.argmax(counts)))
        chosen = list(dict.fromkeys(representatives))
        remaining = [int(z) for z in positive if int(z) not in chosen]
        if max_slices is not None:
            chosen = chosen[:max_slices]
            slots = max(0, max_slices - len(chosen))
            if slots and remaining:
                pick = np.linspace(0, len(remaining) - 1, min(slots, len(remaining))).round().astype(int)
                chosen.extend(remaining[i] for i in np.unique(pick))
            slots = max(0, max_slices - len(chosen))
            if slots and len(negative):
                pick = np.linspace(0, len(negative) - 1, min(slots, len(negative))).round().astype(int)
                chosen.extend(int(negative[i]) for i in np.unique(pick))
        else:
            chosen.extend(remaining)
            chosen.extend(int(z) for z in negative)
        indices = np.asarray(list(dict.fromkeys(chosen)), dtype=int)
    elif max_slices is not None and len(indices) > max_slices:
        pick = np.linspace(0, len(indices) - 1, max_slices).round().astype(int)
        indices = indices[np.unique(pick)]
    if return_indices:
        return [(int(z), m[:, z], y[z]) for z in indices]
    return [(m[:, z], y[z]) for z in indices]
