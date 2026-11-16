from __future__ import annotations
"""Train, select and evaluate the optional native-resolution 2D U-Net on MSD."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is optional. Install it with: python -m pip install -r requirements-unet.txt") from exc

from qmi.data.msd_dataset import read_msd_metadata, discover_msd_cases, load_msd_case
from qmi.data.splits import split_patients
from qmi.metrics import segmentation_metrics, json_safe
from qmi.unet import (UNet2D, choose_device, encode_internal_labels, decode_class_indices,
                      combined_segmentation_loss, select_patient_slice_indices, clear_unet_model_cache)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _load_or_create_split(case_ids: list[str], path: Path, seed: int, new_split: bool) -> dict:
    ids = set(case_ids)
    if path.exists() and not new_split:
        split = json.loads(path.read_text(encoding="utf-8"))
        parts = {name: set(split.get(name, [])) for name in ("train", "validation", "test")}
        union = parts["train"] | parts["validation"] | parts["test"]
        overlap = ((parts["train"] & parts["validation"]) |
                   (parts["train"] & parts["test"]) |
                   (parts["validation"] & parts["test"]))
        if overlap:
            raise RuntimeError(f"Saved patient split contains overlap: {sorted(overlap)[:5]}")
        if union != ids:
            raise RuntimeError(
                "Existing msd_patient_split.json does not exactly cover the current dataset. "
                "Do not silently change your RF/quantum test cohort. Inspect the split, then "
                "rerun with --new-split only if you intentionally want a new shared cohort. "
                f"Missing={len(ids-union)}, unknown={len(union-ids)}"
            )
        if not all(parts[name] for name in parts):
            raise RuntimeError("Saved patient split has an empty train/validation/test partition")
        return split
    if path.exists() and new_split:
        backup = path.with_name(path.stem + "_before_unet_new_split.json")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
    parts = split_patients(case_ids, seed=seed)
    split = {"seed": seed, "selected_case_ids": list(case_ids), **parts}
    _atomic_json(path, split)
    return split


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array, allow_pickle=False)
    tmp.replace(path)


def build_cache(cases_by_id: dict, patient_ids: list[str], metadata, cache_root: Path,
                slices_per_patient: int, rebuild: bool) -> None:
    image_dir, label_dir, index_dir = cache_root / "images", cache_root / "labels", cache_root / "indices"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.json"
    expected_ids = sorted(set(patient_ids))
    manifest_ok = False
    if manifest_path.exists() and not rebuild:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_ok = (manifest.get("version") == 2 and
                           manifest.get("slices_per_patient") == slices_per_patient and
                           manifest.get("patient_ids") == expected_ids)
        except (OSError, json.JSONDecodeError):
            manifest_ok = False
    if manifest_ok and all((image_dir / f"{pid}.npy").is_file() and
                           (label_dir / f"{pid}.npy").is_file() and
                           (index_dir / f"{pid}.npy").is_file() for pid in expected_ids):
        print(f"Using existing U-Net slice cache: {cache_root}")
        return

    print(f"Preparing native 240x240 U-Net cache for {len(expected_ids)} patients...")
    for i, patient_id in enumerate(expected_ids, start=1):
        case = cases_by_id[patient_id]
        modalities, labels = load_msd_case(case, metadata)
        if modalities.shape[-2:] != (240, 240):
            raise ValueError(f"{patient_id}: expected native 240x240 slices; got {modalities.shape[-2:]}")
        indices = select_patient_slice_indices(labels, max_slices=slices_per_patient)
        x = modalities[:, indices].transpose(1, 0, 2, 3).astype(np.float16)
        y = labels[indices].astype(np.uint8)
        _save_npy_atomic(image_dir / f"{patient_id}.npy", x)
        _save_npy_atomic(label_dir / f"{patient_id}.npy", y)
        _save_npy_atomic(index_dir / f"{patient_id}.npy", indices.astype(np.int16))
        if i == 1 or i % 20 == 0 or i == len(expected_ids):
            print(f"  cached {i}/{len(expected_ids)} patients")
        del modalities, labels, x, y
    _atomic_json(manifest_path, {"version": 2, "dataset": "MSD Task01_BrainTumour",
        "slices_per_patient": slices_per_patient, "patient_ids": expected_ids,
        "image_dtype": "float16", "label_dtype": "uint8", "resolution": [240, 240],
        "stores_original_axial_indices": True})


def compute_class_weights(patient_ids: list[str], cache_root: Path, device: str) -> torch.Tensor:
    counts = np.zeros(4, dtype=np.int64)
    for patient_id in patient_ids:
        y = np.load(cache_root / "labels" / f"{patient_id}.npy", mmap_mode="r", allow_pickle=False)
        mapped = encode_internal_labels(y)
        counts += np.bincount(mapped.ravel(), minlength=4)
    weights = np.sqrt(counts.sum() / (4.0 * np.maximum(counts, 1)))
    weights = np.clip(weights, 0.5, 5.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _patient_batches(patient_ids: list[str], cache_root: Path, batch_size: int,
                     rng: np.random.Generator, shuffle: bool):
    order = list(patient_ids)
    if shuffle:
        rng.shuffle(order)
    for patient_id in order:
        images = np.load(cache_root / "images" / f"{patient_id}.npy", allow_pickle=False)
        labels = np.load(cache_root / "labels" / f"{patient_id}.npy", allow_pickle=False)
        indices = np.arange(len(labels))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            yield images[idx].astype(np.float32), labels[idx]
        del images, labels


def augment_batch(images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Light, label-preserving 2D augmentation applied identically to all modalities."""
    if random.random() < 0.5:
        images, targets = torch.flip(images, (-1,)), torch.flip(targets, (-1,))
    if random.random() < 0.5:
        images, targets = torch.flip(images, (-2,)), torch.flip(targets, (-2,))
    k = random.randrange(4)
    if k:
        images, targets = torch.rot90(images, k, (-2, -1)), torch.rot90(targets, k, (-2, -1))
    scale = torch.empty((images.shape[0], 4, 1, 1), device=images.device).uniform_(0.92, 1.08)
    shift = torch.empty((images.shape[0], 4, 1, 1), device=images.device).uniform_(-0.035, 0.035)
    images = (images * scale + shift).clamp_(0.0, 1.0)
    if random.random() < 0.35:
        images = (images + torch.randn_like(images) * 0.012).clamp_(0.0, 1.0)
    return images, targets


def _predict_indices(model, xb: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        return torch.argmax(model(xb), dim=1).cpu().numpy().astype(np.uint8)


def validation_active_dice(model, patient_ids: list[str], cache_root: Path, device: str,
                           batch_size: int) -> float:
    """Fast validation-only active-region Dice; avoids HD95 distance transforms per epoch."""
    model.eval()
    patient_scores = []
    region_ids = ((1, 2, 3), (1, 3), (3,))  # WT, TC, ET in contiguous class indices
    with torch.inference_mode():
        for patient_id in patient_ids:
            images = np.load(cache_root / "images" / f"{patient_id}.npy", allow_pickle=False)
            labels = np.load(cache_root / "labels" / f"{patient_id}.npy", allow_pickle=False)
            slice_scores = []
            for start in range(0, len(labels), batch_size):
                xb = torch.from_numpy(images[start:start + batch_size].astype(np.float32)).to(device)
                pred = torch.argmax(model(xb), dim=1).cpu().numpy()
                truth = encode_internal_labels(labels[start:start + batch_size])
                for p, t in zip(pred, truth):
                    region_dices = []
                    for classes in region_ids:
                        pm = np.isin(p, classes)
                        tm = np.isin(t, classes)
                        denom = int(pm.sum()) + int(tm.sum())
                        if denom == 0:
                            continue
                        intersection = int(np.count_nonzero(pm & tm))
                        region_dices.append((2.0 * intersection) / denom)
                    if region_dices:
                        slice_scores.append(float(np.mean(region_dices)))
            if slice_scores:
                patient_scores.append(float(np.mean(slice_scores)))
            del images, labels
    return float(np.mean(patient_scores)) if patient_scores else 0.0


def evaluate_split(model, patient_ids: list[str], cache_root: Path, device: str,
                   batch_size: int, split_name: str) -> tuple[pd.DataFrame, dict]:
    model.eval()
    rows = []
    for patient_id in patient_ids:
        images = np.load(cache_root / "images" / f"{patient_id}.npy", allow_pickle=False)
        labels = np.load(cache_root / "labels" / f"{patient_id}.npy", allow_pickle=False)
        source_indices = np.load(cache_root / "indices" / f"{patient_id}.npy", allow_pickle=False)
        for start in range(0, len(labels), batch_size):
            xb = torch.from_numpy(images[start:start+batch_size].astype(np.float32)).to(device)
            pred_idx = _predict_indices(model, xb)
            preds = decode_class_indices(pred_idx)
            for j, pred in enumerate(preds):
                slice_pos = start + j
                truth = labels[slice_pos]
                metrics = segmentation_metrics(pred, truth)
                row = {"sample_id": f"{patient_id}_z{int(source_indices[slice_pos]):03d}", "patient_id": patient_id,
                       "slice_index": int(source_indices[slice_pos]), "slice_index_in_cache": int(slice_pos),
                       "split": split_name, "method": "UNet2D",
                       "resolution": 240, "inference_seconds": None,
                       "macro_dice": metrics["macro_dice"], "macro_iou": metrics["macro_iou"],
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
        del images, labels
    per_case = pd.DataFrame(rows)
    numeric = [c for c in per_case.select_dtypes(include=[np.number]).columns
               if c not in ("resolution", "slice_index", "slice_index_in_cache")]
    summary = per_case.groupby(["method", "resolution"])[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return per_case, summary.reset_index()


def evaluate_paired_rf_comparison(test_ids: list[str], split: dict, cache_root: Path,
                                  artifacts: Path, reports: Path, device: str,
                                  batch_size: int) -> dict:
    """Evaluate RF and U-Net on exactly the same held-out cached slices when lineage matches."""
    rf_path = artifacts / "msd_rf_240.joblib"
    rf_meta_path = artifacts / "msd_rf_240.json"
    if not rf_path.is_file() or not rf_meta_path.is_file():
        return {"available": False, "reason": "Native-resolution RF artifact or metadata is missing."}
    try:
        rf_meta = json.loads(rf_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"Could not read RF metadata: {exc}"}
    rf_splits = rf_meta.get("patient_splits")
    if not isinstance(rf_splits, dict):
        return {"available": False, "reason": "RF artifact metadata does not record patient_splits."}
    for name in ("train", "validation", "test"):
        if set(rf_splits.get(name, [])) != set(split.get(name, [])):
            return {"available": False, "reason": f"RF and U-Net {name} patient partitions differ; paired comparison skipped."}
    try:
        from qmi.models import load_model, predict_slice
        rf_model, _ = load_model(rf_path)
    except Exception as exc:
        return {"available": False, "reason": f"Could not load the native RF model: {type(exc).__name__}: {exc}"}

    checkpoint_path = artifacts / "msd_unet_240.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    unet_cfg = checkpoint.get("model_config", {})
    unet_model = UNet2D(base_channels=int(unet_cfg.get("base_channels", 16)),
                        dropout=float(unet_cfg.get("dropout", 0.08))).to(device)
    unet_model.load_state_dict(checkpoint["state_dict"])
    unet_model.eval()
    rows = []
    for patient_id in test_ids:
        images = np.load(cache_root / "images" / f"{patient_id}.npy", allow_pickle=False)
        labels = np.load(cache_root / "labels" / f"{patient_id}.npy", allow_pickle=False)
        indices = np.load(cache_root / "indices" / f"{patient_id}.npy", allow_pickle=False)
        for start in range(0, len(labels), batch_size):
            image_batch = images[start:start + batch_size].astype(np.float32)
            xb = torch.from_numpy(image_batch).to(device)
            unet_indices = _predict_indices(unet_model, xb)
            unet_predictions = decode_class_indices(unet_indices)
            for offset, image in enumerate(image_batch):
                cache_index = start + offset
                truth = labels[cache_index]
                rf_prediction = predict_slice(rf_model, image)
                for method, prediction in (("RandomForest", rf_prediction), ("Residual2DUNet", unet_predictions[offset])):
                    metrics = segmentation_metrics(prediction, truth)
                    row = {"patient_id": patient_id, "slice_index": int(indices[cache_index]),
                           "sample_id": f"{patient_id}_z{int(indices[cache_index]):03d}",
                           "method": method, "resolution": 240,
                           "macro_dice": metrics["macro_dice"], "macro_iou": metrics["macro_iou"],
                           "macro_dice_active": metrics["macro_dice_active"],
                           "macro_iou_active": metrics["macro_iou_active"]}
                    for region in ("WT", "TC", "ET"):
                        row[f"{region}_dice"] = metrics[region]["dice"]
                        row[f"{region}_iou"] = metrics[region]["iou"]
                        row[f"{region}_precision"] = metrics[region]["precision"]
                        row[f"{region}_recall"] = metrics[region]["recall"]
                        row[f"{region}_hd95_pixels"] = metrics[region]["hd95_pixels"]
                    rows.append(row)
        del images, labels, indices
    paired = pd.DataFrame(rows)
    paired.to_csv(reports / "msd_unet_vs_rf_paired_test_per_slice.csv", index=False)
    numeric = [c for c in paired.select_dtypes(include=[np.number]).columns if c not in ("resolution", "slice_index")]
    summary = paired.groupby("method")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(reports / "msd_unet_vs_rf_paired_test_summary.csv", index=False)
    patient_numeric = [c for c in paired.select_dtypes(include=[np.number]).columns
                       if c not in ("resolution", "slice_index")]
    patient_summary = paired.groupby(["patient_id", "method"])[patient_numeric].mean(numeric_only=True).reset_index()
    patient_summary.to_csv(reports / "msd_unet_vs_rf_paired_test_per_patient.csv", index=False)
    return {"available": True, "same_patient_split": True,
            "same_test_slices": True, "test_patients": len(test_ids),
            "slices": int(paired["sample_id"].nunique()),
            "summary_csv": str(reports / "msd_unet_vs_rf_paired_test_summary.csv"),
            "per_slice_csv": str(reports / "msd_unet_vs_rf_paired_test_per_slice.csv"),
            "per_patient_csv": str(reports / "msd_unet_vs_rf_paired_test_per_patient.csv"),
            "summary": summary.to_dict(orient="records")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a native 240x240 multimodal 2D U-Net for MSD brain-tumour segmentation")
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "Task01_BrainTumour"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--slices-per-patient", type=int, default=12)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--new-split", action="store_true", help="Back up and intentionally regenerate the shared patient split")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "processed" / "unet_cache"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.slices_per_patient, args.patience) < 1:
        parser.error("epochs, batch-size, slices-per-patient and patience must be positive")
    if args.base_channels < 8 or args.learning_rate <= 0:
        parser.error("base-channels must be >=8 and learning-rate must be positive")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(8, (torch.get_num_threads() or 1))))
    device = choose_device(args.device)
    data_root = Path(args.data).expanduser().resolve()
    metadata = read_msd_metadata(data_root)
    cases = discover_msd_cases(data_root)
    if len(cases) < 10:
        raise RuntimeError(f"At least 10 labelled patients are required; found {len(cases)}")
    cases_by_id = {case.case_id: case for case in cases}
    reports = ROOT / "outputs" / "reports"; artifacts = ROOT / "artifacts"
    reports.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
    split_path = reports / "msd_patient_split.json"
    split = _load_or_create_split(list(cases_by_id), split_path, args.seed, args.new_split)
    cache_root = Path(args.cache_dir).expanduser().resolve()
    build_cache(cases_by_id, list(cases_by_id), metadata, cache_root, args.slices_per_patient, args.rebuild_cache)

    train_ids = list(split["train"]); val_ids = list(split["validation"]); test_ids = list(split["test"])
    class_weights = compute_class_weights(train_ids, cache_root, device)
    print(f"Device: {device} | patients: train={len(train_ids)}, validation={len(val_ids)}, test={len(test_ids)}")
    print("Class weights [background, non-enhancing, edema, enhancing]:", class_weights.detach().cpu().numpy().round(3).tolist())
    model = UNet2D(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5)
    rng = np.random.default_rng(args.seed)
    history = []
    best_score = -1.0; best_epoch = 0; stale = 0
    checkpoint_path = artifacts / "msd_unet_240.pt"
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []; epoch_start = time.perf_counter()
        for images_np, labels_np in _patient_batches(train_ids, cache_root, args.batch_size, rng, True):
            xb = torch.from_numpy(images_np).to(device)
            y_idx = torch.from_numpy(encode_internal_labels(labels_np).astype(np.int64)).to(device)
            xb, y_idx = augment_batch(xb, y_idx)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = combined_segmentation_loss(logits, y_idx, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_active = validation_active_dice(model, val_ids, cache_root, device, args.batch_size)
        scheduler.step(val_active)
        epoch_row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                     "validation_active_macro_dice": val_active,
                     "learning_rate": float(optimizer.param_groups[0]["lr"]),
                     "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(epoch_row)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={epoch_row['train_loss']:.4f} | "
              f"val active Dice={val_active:.4f} | {epoch_row['epoch_seconds']:.1f}s")
        if val_active > best_score + 1e-5:
            best_score = val_active; best_epoch = epoch; stale = 0
            split_fingerprint_payload = "\n".join(
                f"{name}:" + ",".join(sorted(split[name])) for name in ("train", "validation", "test"))
            split_fingerprint = hashlib.sha256(split_fingerprint_payload.encode("utf-8")).hexdigest()
            metadata_payload = {"dataset": "Medical Segmentation Decathlon Task01_BrainTumour",
                "source_lineage": "BraTS 2016/2017-derived; not BraTS 2021",
                "model_family": "residual 2D U-Net", "task": "native-resolution 2D axial segmentation",
                "label_ids": [0, 1, 2, 4], "regions": {"WT": [1, 2, 4], "TC": [1, 4], "ET": [4]},
                "best_epoch": best_epoch, "best_validation_active_macro_dice": best_score,
                "patient_split": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
                "patient_split_sha256": split_fingerprint,
                "split_seed": int(split.get("seed", args.seed)), "training_config": vars(args)}
            checkpoint_tmp = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({"state_dict": model.state_dict(),
                        "model_config": {"base_channels": args.base_channels, "dropout": 0.08},
                        "metadata": metadata_payload}, checkpoint_tmp)
            checkpoint_tmp.replace(checkpoint_path)
            _atomic_json(artifacts / "msd_unet_240.json", metadata_payload)
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement.")
            break

    if not checkpoint_path.is_file():
        raise RuntimeError("Training ended without a best-validation checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    # Test patients are touched only after validation-based checkpoint selection is complete.
    val_case, val_summary = evaluate_split(model, val_ids, cache_root, device, args.batch_size, "validation")
    test_case, test_summary = evaluate_split(model, test_ids, cache_root, device, args.batch_size, "test")
    val_case.to_csv(reports / "msd_unet_validation_per_slice.csv", index=False)
    val_summary.to_csv(reports / "msd_unet_validation_summary.csv", index=False)
    test_case.to_csv(reports / "msd_unet_test_per_slice.csv", index=False)
    test_summary.to_csv(reports / "msd_unet_test_summary.csv", index=False)
    numeric = [c for c in test_case.select_dtypes(include=[np.number]).columns
               if c not in ("resolution", "slice_index", "slice_index_in_cache")]
    patient_summary = test_case.groupby(["patient_id", "method"], dropna=False)[numeric].mean(numeric_only=True).reset_index()
    patient_summary.to_csv(reports / "msd_unet_test_per_patient.csv", index=False)
    pd.DataFrame(history).to_csv(reports / "msd_unet_training_history.csv", index=False)
    try:
        paired_comparison = evaluate_paired_rf_comparison(test_ids, split, cache_root, artifacts,
                                                           reports, device, args.batch_size)
    except Exception as exc:
        paired_comparison = {"available": False,
                             "reason": f"Paired RF comparison failed safely: {type(exc).__name__}: {exc}"}
    if paired_comparison.get("available"):
        print("\nPaired held-out comparison (same test patients and exact cached slices):")
        print(pd.read_csv(paired_comparison["summary_csv"]).to_string(index=False))
    else:
        print("\nPaired RF comparison skipped:", paired_comparison.get("reason"))

    summary = {"dataset": "Medical Segmentation Decathlon Task01_BrainTumour",
        "model": "residual 2D U-Net", "resolution": 240, "device": device,
        "best_epoch": best_epoch, "best_validation_active_macro_dice": best_score,
        "epochs_completed": len(history), "wall_seconds": time.perf_counter() - start_time,
        "train_patients": len(train_ids), "validation_patients": len(val_ids), "test_patients": len(test_ids),
        "validation_summary": val_summary.to_dict(orient="records"),
        "heldout_test_summary": test_summary.to_dict(orient="records"),
        "test_patients_used_for_selection": False,
        "paired_rf_comparison": paired_comparison,
        "slice_selection": {"slices_per_patient": args.slices_per_patient,
                            "positive_fraction_target": 0.75,
                            "selection_policy": "representative WT/TC/ET slices, sampled tumour-positive and background slices"},
        "checkpoint": str(checkpoint_path)}
    _atomic_json(reports / "msd_unet_experiment_summary.json", summary)
    print("\nHeld-out U-Net test summary:")
    print(test_summary.to_string(index=False))
    print(f"\nSaved best model: {checkpoint_path}")
    print(f"Saved reports: {reports / 'msd_unet_test_summary.csv'}")
    print("IMPORTANT: compare against RF on the exact same saved patients/slices; this is not clinical validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
