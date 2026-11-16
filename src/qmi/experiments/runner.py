"""One-command, patient-level MSD training, validation and held-out testing."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from ..data.msd_dataset import read_msd_metadata, discover_msd_cases
from ..data.splits import split_patients
from ..training import prepare_msd_patient_slices, iter_msd_patient_slices, train_msd_random_forest
from ..baselines import KMeansMultiClassBaseline, WatershedMultiClassBaseline
from .pipeline import evaluate_methods
from ..db import add_experiment
from ..metrics import json_safe


def run_msd_experiment(dataset_root: str | Path, *, max_patients: int = 120,
                       max_slices_per_patient: int = 8, samples_per_class: int = 250,
                       trees: int = 120, seed: int = 42,
                       resolutions: tuple[int, ...] = (16, 32), output_root: str | Path = ".") -> dict:
    root = Path(output_root).resolve()
    reports = root / "outputs" / "reports"
    artifacts = root / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    metadata = read_msd_metadata(dataset_root)
    cases = discover_msd_cases(dataset_root)
    if len(cases) < 10:
        raise RuntimeError(f"At least 10 labelled patient cases are required; found {len(cases)}")
    if max_patients < 0:
        raise ValueError("max_patients must be 0 (all cases) or a positive integer")
    if max_patients and max_patients < len(cases):
        rng = np.random.default_rng(seed)
        selected_indices = np.sort(rng.choice(len(cases), size=max_patients, replace=False))
        cases = [cases[int(i)] for i in selected_indices]
    splits = split_patients([case.case_id for case in cases], seed=seed)
    selected_ids = [case.case_id for case in cases]
    (reports / "msd_patient_split.json").write_text(json.dumps({"seed": seed, "selected_case_ids": selected_ids, **splits}, indent=2), encoding="utf-8")
    summaries: dict[str, dict] = {}
    for resolution in resolutions:
        if resolution not in (16, 32, 240):
            raise ValueError("Supported resolutions are 16, 32 and native 240")
        t0 = time.perf_counter()
        train_slices = prepare_msd_patient_slices(cases, splits["train"], metadata,
            resolution=resolution, max_slices_per_patient=max_slices_per_patient)
        model_path = artifacts / f"msd_rf_{resolution}.joblib"
        model, training_report = train_msd_random_forest(train_slices, resolution=resolution,
            samples_per_class=samples_per_class, n_estimators=trees, random_state=seed,
            output_path=model_path, splits=splits)
        kmeans = KMeansMultiClassBaseline(random_state=seed).fit([(m, y) for _, m, y in train_slices])
        joblib.dump(kmeans, artifacts / f"msd_kmeans_{resolution}.joblib", compress=3)
        baselines = {"kmeans": kmeans, "watershed": WatershedMultiClassBaseline(kmeans)}
        train_slice_count = len(train_slices)
        del train_slices  # free selected training images before streaming validation/test
        val_iter = iter_msd_patient_slices(cases, splits["validation"], metadata,
            resolution=resolution, max_slices_per_patient=max_slices_per_patient)
        test_iter = iter_msd_patient_slices(cases, splits["test"], metadata,
            resolution=resolution, max_slices_per_patient=max_slices_per_patient)
        val_case, val_summary = evaluate_methods(val_iter, model, baselines, resolution)
        test_case, test_summary = evaluate_methods(test_iter, model, baselines, resolution)
        val_slice_count = int(val_case["sample_id"].nunique())
        test_slice_count = int(test_case["sample_id"].nunique())
        val_case.to_csv(reports / f"msd_validation_per_slice_{resolution}.csv", index=False)
        val_summary.to_csv(reports / f"msd_validation_summary_{resolution}.csv", index=False)
        test_case.to_csv(reports / f"msd_test_per_slice_{resolution}.csv", index=False)
        test_summary.to_csv(reports / f"msd_test_summary_{resolution}.csv", index=False)
        # Patient-level aggregation prevents slices from dominating the reported cohort summary.
        numeric_columns = [c for c in test_case.select_dtypes(include=[np.number]).columns
                           if c not in ("resolution", "slice_index")]
        patient_summary = (test_case.groupby(["patient_id", "method"], dropna=False)[numeric_columns]
                           .mean(numeric_only=True).reset_index())
        patient_summary.to_csv(reports / f"msd_test_per_patient_{resolution}.csv", index=False)
        metric_snapshot = test_summary.to_dict(orient="records")
        run_config = {"resolution": resolution, "seed": seed, "selected_patients": len(cases),
            "train_patients": len(splits["train"]), "validation_patients": len(splits["validation"]),
            "test_patients": len(splits["test"]), "train_slices": train_slice_count,
            "validation_slices": val_slice_count, "test_slices": test_slice_count,
            "samples_per_class_per_slice": samples_per_class, "trees": trees,
            "max_slices_per_patient": max_slices_per_patient}
        elapsed = time.perf_counter() - t0
        model_metadata = {"dataset": "Medical Segmentation Decathlon Task01_BrainTumour",
            "source_lineage": "BraTS 2016/2017-derived; not BraTS 2021", "resolution": resolution,
            "training_report": training_report.__dict__, "patient_splits": splits, "config": run_config}
        (artifacts / f"msd_rf_{resolution}.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
        add_experiment(f"MSD RF + classical baselines ({resolution}x{resolution})",
            "Medical Segmentation Decathlon Task01_BrainTumour", config=run_config,
            metrics={"validation_summary": val_summary.to_dict(orient="records"), "test_summary": metric_snapshot,
                     "training_report": training_report.__dict__, "wall_seconds": elapsed},
            artifact_dir=str(artifacts))
        summaries[str(resolution)] = {"test_summary": metric_snapshot,
            "validation_summary": val_summary.to_dict(orient="records"),
            "training_report": training_report.__dict__, "wall_seconds": elapsed,
            "model_path": str(model_path)}
        print(f"\n[{resolution}x{resolution if resolution != 240 else 240}] {train_slice_count} training slices; "
              f"{val_slice_count} validation; {test_slice_count} held-out test; {elapsed:.1f}s")
        print(test_summary.to_string(index=False))
    result = {"dataset": "Medical Segmentation Decathlon Task01_BrainTumour",
        "source_lineage": "BraTS 2016/2017-derived; not BraTS 2021", "patient_splits": splits,
        "selected_patients": len(cases), "resolutions": summaries}
    (reports / "msd_experiment_summary.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    return result
