"""Controlled comparison of classical baselines and the trained RF model."""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from ..baselines import otsu_multiclass, KMeansMultiClassBaseline, WatershedMultiClassBaseline
from ..data.preprocessing import resize_slice
from ..metrics import segmentation_metrics
from ..models import predict_slice


def split_ids(ids, train_fraction=0.70, validation_fraction=0.15, seed=42):
    ids = np.asarray(list(ids), dtype=object)
    if len(ids) < 3:
        raise ValueError("At least three independent subjects/samples are required")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("Invalid split fractions")
    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(round(len(ids) * train_fraction)))
    n_val = max(1, int(round(len(ids) * validation_fraction)))
    if n_train + n_val >= len(ids):
        n_train, n_val = len(ids)-2, 1
    return {"train": shuffled[:n_train].tolist(), "validation": shuffled[n_train:n_train+n_val].tolist(),
            "test": shuffled[n_train+n_val:].tolist()}


def fit_classical_baselines(training_slices):
    kmeans = KMeansMultiClassBaseline(random_state=42).fit(training_slices)
    watershed = WatershedMultiClassBaseline(kmeans)
    return {"kmeans": kmeans, "watershed": watershed}


def evaluate_methods(test_slices, rf_model, baselines, resolution: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    methods = ["RandomForest", "Otsu", "KMeans", "Watershed"]
    for sample_id, modalities, labels in test_slices:
        m, y = resize_slice(modalities, labels, resolution)
        start = time.perf_counter(); pred_rf = predict_slice(rf_model, m); rf_seconds = time.perf_counter()-start
        predictions = {"RandomForest": (pred_rf, rf_seconds)}
        start = time.perf_counter(); pred = otsu_multiclass(m); predictions["Otsu"] = (pred, time.perf_counter()-start)
        start = time.perf_counter(); pred = baselines["kmeans"].predict(m); predictions["KMeans"] = (pred, time.perf_counter()-start)
        start = time.perf_counter(); pred = baselines["watershed"].predict(m); predictions["Watershed"] = (pred, time.perf_counter()-start)
        for method, (pred, elapsed) in predictions.items():
            metrics = segmentation_metrics(pred, y)
            patient_id, _, slice_text = str(sample_id).partition("_z")
            row = {"sample_id": str(sample_id), "patient_id": patient_id,
                   "slice_index": int(slice_text) if slice_text.isdigit() else None,
                   "method": method, "resolution": resolution,
                   "inference_seconds": float(elapsed), "macro_dice": metrics["macro_dice"],
                   "macro_iou": metrics["macro_iou"],
                   "macro_dice_present_gt": metrics["macro_dice_present_gt"],
                   "macro_iou_present_gt": metrics["macro_iou_present_gt"],
                   "macro_dice_active": metrics["macro_dice_active"],
                   "macro_iou_active": metrics["macro_iou_active"],
                   "n_regions_present_gt": metrics["n_regions_present_gt"],
                   "n_regions_empty_gt": metrics["n_regions_empty_gt"],
                   "n_regions_empty_prediction": metrics["n_regions_empty_prediction"],
                   "n_regions_both_empty": metrics["n_regions_both_empty"]}
            for region in ("WT", "TC", "ET"):
                for metric in ("dice", "iou", "precision", "recall", "specificity", "hd95_pixels",
                               "tp", "fp", "fn", "tn", "empty_prediction", "empty_ground_truth"):
                    row[f"{region}_{metric}"] = metrics[region][metric]
            rows.append(row)
    per_case = pd.DataFrame(rows)
    metric_cols = [c for c in per_case.columns if c not in ("sample_id", "patient_id", "slice_index", "method", "resolution")]
    summary = per_case.groupby(["method", "resolution"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    return per_case, summary.reset_index()


def save_evaluation(per_case: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(out / "per_case_metrics.csv", index=False)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    (out / "evaluation_metadata.json").write_text(json.dumps({"note": "Metrics are data-dependent; synthetic demo results are not clinical or BraTS results."}, indent=2), encoding="utf-8")
