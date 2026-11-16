"""Reconstruction and region-wise segmentation metrics with explicit empty-mask semantics."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.metrics import structural_similarity

REGIONS = {
    "WT": lambda y: np.isin(y, [1, 2, 4]),
    "TC": lambda y: np.isin(y, [1, 4]),
    "ET": lambda y: y == 4,
}


def reconstruction_metrics(reference: np.ndarray, reconstructed: np.ndarray, data_range: float | None = None) -> dict:
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(reconstructed, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("Reference and reconstructed images must have identical shapes")
    if a.ndim != 2:
        raise ValueError("Reconstruction metrics expect 2D grayscale images")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Images contain non-finite values")
    if data_range is None:
        data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
        if data_range <= 0:
            data_range = 1.0
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be a positive finite number")
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10((data_range ** 2) / mse))
    min_dim = min(a.shape)
    win_size = min(7, min_dim if min_dim % 2 else min_dim - 1)
    ssim = float("nan") if win_size < 3 else float(structural_similarity(a, b, data_range=data_range, win_size=win_size))
    return {"mse": mse, "psnr_db": psnr, "ssim": ssim}


def _binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    p_arr = np.asarray(pred, dtype=bool)
    g_arr = np.asarray(truth, dtype=bool)
    if p_arr.shape != g_arr.shape:
        raise ValueError("Prediction and ground truth masks must have identical shapes")
    p = p_arr.ravel()
    g = g_arr.ravel()
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(np.count_nonzero(~p & ~g))
    pred_empty = not bool(np.any(p))
    truth_empty = not bool(np.any(g))
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    dice = 1.0 if denom_dice == 0 else (2.0 * tp / denom_dice)
    iou = 1.0 if denom_iou == 0 else (tp / denom_iou)
    # A metric with a zero denominator is undefined, not a perfect score.
    precision = float("nan") if tp + fp == 0 else tp / (tp + fp)
    recall = float("nan") if tp + fn == 0 else tp / (tp + fn)
    specificity = float("nan") if tn + fp == 0 else tn / (tn + fp)
    if pred_empty or truth_empty:
        # HD95 is undefined if either surface is absent, including empty/empty.
        hd95 = float("nan")
    else:
        p_surface = p_arr & ~binary_erosion(p_arr, border_value=0)
        g_surface = g_arr & ~binary_erosion(g_arr, border_value=0)
        d_to_g = distance_transform_edt(~g_surface)
        d_to_p = distance_transform_edt(~p_surface)
        # Use the maximum of the two directed 95th-percentile surface distances.
        # Pooling both directions can hide a large one-way boundary error when
        # the predicted and reference surfaces contain very different numbers
        # of pixels.
        directed_pred_to_truth = d_to_g[p_surface]
        directed_truth_to_pred = d_to_p[g_surface]
        if directed_pred_to_truth.size and directed_truth_to_pred.size:
            hd95 = float(max(np.percentile(directed_pred_to_truth, 95),
                             np.percentile(directed_truth_to_pred, 95)))
        else:
            hd95 = float("nan")
    return {
        "dice": float(dice), "iou": float(iou), "precision": float(precision),
        "recall": float(recall), "sensitivity": float(recall), "specificity": float(specificity),
        "hd95_pixels": float(hd95), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "empty_prediction": pred_empty, "empty_ground_truth": truth_empty,
    }


def segmentation_metrics(pred_labels: np.ndarray, true_labels: np.ndarray) -> dict:
    pred = np.asarray(pred_labels)
    truth = np.asarray(true_labels)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and ground-truth label maps must have identical shapes")
    if pred.ndim != 2:
        raise ValueError("Segmentation metrics expect 2D label maps")
    if not np.all(np.isin(pred, [0, 1, 2, 4])) or not np.all(np.isin(truth, [0, 1, 2, 4])):
        raise ValueError("Label maps must use BraTS labels {0,1,2,4}")
    result = {name: _binary_metrics(fn(pred), fn(truth)) for name, fn in REGIONS.items()}
    all_dice = [result[k]["dice"] for k in REGIONS]
    all_iou = [result[k]["iou"] for k in REGIONS]
    present_gt = [k for k in REGIONS if not result[k]["empty_ground_truth"]]
    active = [k for k in REGIONS if not (result[k]["empty_ground_truth"] and result[k]["empty_prediction"])]
    result["macro_dice"] = float(np.mean(all_dice))
    result["macro_iou"] = float(np.mean(all_iou))
    result["macro_dice_present_gt"] = float(np.mean([result[k]["dice"] for k in present_gt])) if present_gt else float("nan")
    result["macro_iou_present_gt"] = float(np.mean([result[k]["iou"] for k in present_gt])) if present_gt else float("nan")
    result["macro_dice_active"] = float(np.mean([result[k]["dice"] for k in active])) if active else float("nan")
    result["macro_iou_active"] = float(np.mean([result[k]["iou"] for k in active])) if active else float("nan")
    result["n_regions_present_gt"] = int(len(present_gt))
    result["n_regions_empty_gt"] = int(sum(result[k]["empty_ground_truth"] for k in REGIONS))
    result["n_regions_empty_prediction"] = int(sum(result[k]["empty_prediction"] for k in REGIONS))
    result["n_regions_both_empty"] = int(sum(result[k]["empty_ground_truth"] and result[k]["empty_prediction"] for k in REGIONS))
    return result


def json_safe(value):
    """Recursively replace NaN/Inf with None for strict JSON APIs and manifests."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
