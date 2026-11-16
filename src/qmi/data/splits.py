"""Patient-level train/validation/test splitting."""
from __future__ import annotations
import numpy as np


def split_patients(patient_ids, train_fraction=0.70, validation_fraction=0.15, seed=42):
    ids = np.asarray(list(patient_ids), dtype=object)
    if len(ids) < 3:
        raise ValueError("At least three patients are required for independent train/validation/test splits")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Patient IDs must be unique")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("Invalid split fractions")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = max(1, int(np.floor(len(ids) * train_fraction)))
    n_val = max(1, int(np.floor(len(ids) * validation_fraction)))
    if n_train + n_val >= len(ids):
        n_train, n_val = len(ids)-2, 1
    result = {"train": ids[:n_train].tolist(), "validation": ids[n_train:n_train+n_val].tolist(),
              "test": ids[n_train+n_val:].tolist()}
    if set(result["train"]) & set(result["validation"]) or set(result["train"]) & set(result["test"]) or set(result["validation"]) & set(result["test"]):
        raise RuntimeError("Patient leakage detected during split")
    return result
