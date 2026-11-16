"""MSD Task01 patient-level preparation and classical model training."""
from __future__ import annotations
from pathlib import Path
from .data.msd_dataset import MSDCase, MSDMetadata, load_msd_case, case_to_slices
from .data.preprocessing import resize_slice
from .data.splits import split_patients
from .models import balanced_pixel_samples, train_random_forest, save_model


def iter_msd_patient_slices(cases: list[MSDCase], patient_ids: list[str], metadata: MSDMetadata,
                            *, resolution: int = 32, max_slices_per_patient: int | None = 8):
    """Yield selected slices one patient at a time to bound validation/test memory."""
    by_id = {case.case_id: case for case in cases}
    missing = sorted(set(patient_ids) - set(by_id))
    if missing:
        raise KeyError(f"Unknown patient IDs: {missing[:5]}")
    for patient_id in patient_ids:
        modalities, labels = load_msd_case(by_id[patient_id], metadata)
        selected = case_to_slices(modalities, labels, max_slices=max_slices_per_patient,
                                  tumor_first=True, return_indices=True)
        for z_index, image_slice, label_slice in selected:
            m_small, y_small = resize_slice(image_slice, label_slice, resolution)
            yield (f"{patient_id}_z{z_index:03d}", m_small, y_small)
        # Release full patient volumes before loading the next subject.
        del modalities, labels, selected


def prepare_msd_patient_slices(cases: list[MSDCase], patient_ids: list[str], metadata: MSDMetadata,
                               *, resolution: int = 32, max_slices_per_patient: int | None = 8):
    """Materialize training slices; use the iterator for validation/test to bound memory."""
    return list(iter_msd_patient_slices(cases, patient_ids, metadata, resolution=resolution,
                                         max_slices_per_patient=max_slices_per_patient))


def train_msd_random_forest(train_slices, *, resolution: int, samples_per_class: int = 250,
                             n_estimators: int = 120, max_depth: int = 18, random_state: int = 42,
                             output_path: str | Path = "artifacts/msd_rf.joblib", splits: dict | None = None):
    X, y = balanced_pixel_samples([(m, lab) for _, m, lab in train_slices],
                                  samples_per_class=samples_per_class, random_state=random_state)
    model, report = train_random_forest(X, y, n_estimators=n_estimators, max_depth=max_depth,
                                        random_state=random_state)
    save_model(model, output_path, metadata={
        "dataset": "Medical Segmentation Decathlon Task01_BrainTumour",
        "source_lineage": "BraTS 2016/2017-derived; not BraTS 2021",
        "resolution": int(resolution), "random_state": int(random_state),
        "patient_splits": splits or {}, "training_report": report.__dict__,
        "input_modalities": ["FLAIR", "T1", "T1ce", "T2"],
        "internal_labels": {"0": "background", "1": "non-enhancing tumour", "2": "edema", "4": "enhancing tumour"},
        "regions": {"WT": [1, 2, 4], "TC": [1, 4], "ET": [4]},
    })
    return model, report
