"""Legacy BraTS-style NIfTI reader retained for compatibility tests only.

The active project pipeline reads MSD Task01 natively via msd_dataset.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class BratsCase:
    case_id: str
    flair: Path
    t1: Path
    t1ce: Path
    t2: Path
    seg: Path


def discover_brats_cases(root: str | Path) -> list[BratsCase]:
    """Find BraTS-style cases recursively; supports files in case folders or a flat folder."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []
    cases: list[BratsCase] = []
    for flair in sorted(root.rglob("*_flair.nii*")):
        stem = flair.name
        suffix = "_flair.nii.gz" if stem.endswith("_flair.nii.gz") else "_flair.nii"
        if not stem.endswith(suffix):
            continue
        prefix = stem[:-len(suffix)]
        folder = flair.parent
        paths = {
            "flair": flair,
            "t1": folder / f"{prefix}_t1.nii.gz",
            "t1ce": folder / f"{prefix}_t1ce.nii.gz",
            "t2": folder / f"{prefix}_t2.nii.gz",
            "seg": folder / f"{prefix}_seg.nii.gz",
        }
        # Also accept uncompressed NIfTI companions.
        for key in ("t1", "t1ce", "t2", "seg"):
            if not paths[key].exists():
                paths[key] = folder / f"{prefix}_{key}.nii"
        if all(p.exists() for p in paths.values()):
            cases.append(BratsCase(prefix, paths["flair"], paths["t1"], paths["t1ce"], paths["t2"], paths["seg"]))
    # Avoid duplicate case entries if a directory contains unusual nested symlinks.
    unique = {case.case_id: case for case in cases}
    return [unique[k] for k in sorted(unique)]


def _robust_normalize(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    mask = np.isfinite(arr) & (arr != 0)
    if not np.any(mask):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[mask], [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def load_brats_case(case: BratsCase) -> tuple[np.ndarray, np.ndarray]:
    """Return modalities (4,D,H,W) in FLAIR,T1,T1ce,T2 order and labels (D,H,W)."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("NIfTI loading requires nibabel. Install with: pip install nibabel") from exc
    paths = [case.flair, case.t1, case.t1ce, case.t2, case.seg]
    # Canonicalize every volume to RAS+ before selecting the final (axial) axis.
    images = [nib.as_closest_canonical(nib.load(str(path))) for path in paths]
    reference_affine = np.asarray(images[0].affine, dtype=np.float64)
    if any(not np.allclose(np.asarray(img.affine), reference_affine, rtol=0, atol=1e-3) for img in images[1:]):
        raise ValueError(f"NIfTI affines are not spatially aligned for {case.case_id}")
    loaded = [np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32) for img in images]
    arrays, seg = loaded[:4], loaded[4]
    shape = arrays[0].shape
    if len(shape) != 3 or any(a.shape != shape for a in arrays) or seg.shape != shape:
        raise ValueError(f"Inconsistent or non-3D NIfTI shapes for {case.case_id}: {[a.shape for a in arrays]} + {seg.shape}")
    arrays = [_robust_normalize(a) for a in arrays]
    seg = np.rint(seg).astype(np.uint8)
    allowed = np.isin(seg, [0, 1, 2, 4])
    if not np.all(allowed):
        unknown = np.unique(seg[~allowed]).tolist()
        raise ValueError(f"Unexpected BraTS labels in {case.case_id}: {unknown}")
    # Most BraTS files are (H,W,D); move the final axis to the slice axis.
    modalities = np.stack(arrays, axis=0)
    modalities = np.moveaxis(modalities, -1, 1)  # (4,D,H,W)
    seg = np.moveaxis(seg, -1, 0)                # (D,H,W)
    return modalities.astype(np.float32), seg.astype(np.uint8)


def case_to_slices(modalities: np.ndarray, labels: np.ndarray, max_slices: int | None = None,
                   tumor_first: bool = True, return_indices: bool = False):
    """Convert one loaded volume to 2D axial slices, retaining aligned labels."""
    if modalities.ndim != 4 or labels.ndim != 3 or modalities.shape[1:] != labels.shape:
        raise ValueError("Expected modalities (4,D,H,W) aligned with labels (D,H,W)")
    indices = np.arange(labels.shape[0])
    if tumor_first:
        positive = indices[np.any(labels > 0, axis=(1, 2))]
        negative = indices[~np.isin(indices, positive)]
        # Prefer at least one representative slice containing each observed tumour label.
        representatives = []
        for cls in (1, 2, 4):
            per_slice = np.count_nonzero(labels == cls, axis=(1, 2))
            if np.any(per_slice > 0):
                representatives.append(int(np.argmax(per_slice)))
        representatives = list(dict.fromkeys(representatives))
        positive_remaining = np.array([z for z in positive if int(z) not in representatives], dtype=int)
        if max_slices is not None:
            chosen = representatives[:max_slices]
            remaining_slots = max(0, max_slices - len(chosen))
            if remaining_slots and len(positive_remaining):
                take = min(remaining_slots, len(positive_remaining))
                pick = np.linspace(0, len(positive_remaining)-1, take).round().astype(int)
                chosen.extend(positive_remaining[np.unique(pick)].tolist())
                remaining_slots = max(0, max_slices - len(chosen))
            if remaining_slots and len(negative):
                take = min(remaining_slots, len(negative))
                pick = np.linspace(0, len(negative)-1, take).round().astype(int)
                chosen.extend(negative[np.unique(pick)].tolist())
            indices = np.array(list(dict.fromkeys(chosen)), dtype=int)
        else:
            indices = np.array(representatives + positive_remaining.tolist() + negative.tolist(), dtype=int)
    elif max_slices is not None and len(indices) > max_slices:
        pick = np.linspace(0, len(indices)-1, max_slices).round().astype(int)
        indices = indices[np.unique(pick)]
    if return_indices:
        return [(int(z), modalities[:, int(z)], labels[int(z)]) for z in indices]
    return [(modalities[:, int(z)], labels[int(z)]) for z in indices]
