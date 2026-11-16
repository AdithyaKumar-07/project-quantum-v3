"""Label and modality mapping helpers for Medical Segmentation Decathlon Task01."""
from __future__ import annotations

import re
from typing import Mapping
import numpy as np


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def modality_index_map(modality_metadata: Mapping[str, str]) -> dict[str, int]:
    """Map MSD metadata indices to this project's FLAIR,T1,T1ce,T2 order."""
    result: dict[str, int] = {}
    for raw_index, raw_name in modality_metadata.items():
        name = _normalise(raw_name)
        idx = int(raw_index)
        if "flair" in name:
            key = "flair"
        elif "t1" in name and any(token in name for token in ("gd", "ce", "contrast", "gadolinium", "post")):
            key = "t1ce"
        elif "t1" in name:
            key = "t1"
        elif "t2" in name:
            key = "t2"
        else:
            continue
        if key in result:
            raise ValueError(f"Duplicate modality description for {key!r} in dataset.json")
        result[key] = idx
    required = {"flair", "t1", "t1ce", "t2"}
    if set(result) != required:
        raise ValueError(
            "Could not identify all four MRI modalities from dataset.json. "
            f"Found {sorted(result)}; expected {sorted(required)}."
        )
    if len(set(result.values())) != 4:
        raise ValueError("dataset.json assigns duplicate channel indices to MRI modalities")
    return result


def label_index_map(label_metadata: Mapping[str, str]) -> dict[int, int]:
    """Map MSD label IDs to the project's internal legacy BraTS convention.

    MSD Task01 convention: 0 background, 1 edema, 2 non-enhancing tumour,
    3 enhancing tumour. Internal convention: 0 background, 1 NCR/NET,
    2 edema, 4 enhancing tumour.
    """
    result: dict[int, int] = {}
    for raw_id, raw_name in label_metadata.items():
        source_id = int(raw_id)
        name = _normalise(raw_name)
        if "background" in name:
            target_id = 0
        elif "edema" in name or "oedema" in name:
            target_id = 2
        elif "non enhancing" in name or "nonenhancing" in name or "necrot" in name:
            target_id = 1
        elif "enhanc" in name:
            target_id = 4
        else:
            raise ValueError(f"Unrecognized MSD label description: id={source_id}, name={raw_name!r}")
        if target_id in result.values():
            raise ValueError(f"Multiple source labels map to internal BraTS label {target_id}")
        result[source_id] = target_id
    if result.get(0) != 0 or set(result.values()) != {0, 1, 2, 4}:
        raise ValueError(
            "dataset.json label metadata does not describe background, edema, "
            "non-enhancing tumour and enhancing tumour. Refusing unsafe label conversion."
        )
    return result


def remap_msd_labels(labels: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    """Convert an MSD label volume to internal labels {0,1,2,4}, rejecting unknown IDs."""
    source = np.asarray(labels)
    rounded = np.rint(source).astype(np.int64)
    if not np.allclose(source, rounded, rtol=0, atol=1e-5):
        raise ValueError("Segmentation contains non-integer label values")
    unknown = sorted(set(np.unique(rounded).tolist()) - set(int(k) for k in mapping))
    if unknown:
        raise ValueError(f"Segmentation contains labels absent from dataset.json: {unknown}")
    output = np.zeros(rounded.shape, dtype=np.uint8)
    for src, dst in mapping.items():
        output[rounded == int(src)] = int(dst)
    if not np.all(np.isin(output, [0, 1, 2, 4])):
        raise AssertionError("Internal label conversion produced an unsupported value")
    return output
