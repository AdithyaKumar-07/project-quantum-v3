from __future__ import annotations
"""Fit validation-only tumour thresholds for existing MSD RF models and refresh reports."""
import argparse
import json
import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmi.data.msd_dataset import read_msd_metadata, discover_msd_cases
from qmi.training import iter_msd_patient_slices
from qmi.models import load_model, save_model
from qmi.calibration import calibrate_tumour_threshold
from qmi.baselines import WatershedMultiClassBaseline
from qmi.experiments.pipeline import evaluate_methods
from qmi.experiments.resolutions import parse_resolutions


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate trained MSD RF models using validation patients only")
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "Task01_BrainTumour"))
    parser.add_argument("--resolutions", choices=["16", "32", "240", "both", "all"], default="all")
    parser.add_argument("--slices-per-patient", type=int, default=8)
    args = parser.parse_args()
    resolutions = parse_resolutions(args.resolutions)
    if args.slices_per_patient < 1:
        parser.error("--slices-per-patient must be >= 1")

    data_root = Path(args.data).resolve()
    metadata = read_msd_metadata(data_root)
    cases = discover_msd_cases(data_root)
    split_path = ROOT / "outputs" / "reports" / "msd_patient_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Patient split not found: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    case_ids = {c.case_id for c in cases}
    for part in ("train", "validation", "test"):
        absent = sorted(set(split.get(part, [])) - case_ids)
        if absent:
            raise RuntimeError(f"Saved {part} split contains unavailable cases: {absent[:5]}")
    if not split.get("validation") or not split.get("test"):
        raise RuntimeError("Both validation and held-out test patient partitions are required")

    reports = ROOT / "outputs" / "reports"
    artifacts = ROOT / "artifacts"
    summary = {"dataset": "Medical Segmentation Decathlon Task01_BrainTumour",
               "calibration_source": "validation patients only", "test_source": "held-out test patients only",
               "resolutions": {}}
    for resolution in resolutions:
        model_path = artifacts / f"msd_rf_{resolution}.joblib"
        baseline_path = artifacts / f"msd_kmeans_{resolution}.joblib"
        if not model_path.is_file() or not baseline_path.is_file():
            raise FileNotFoundError(f"Missing model/baseline for {resolution}; run full-cohort training first")
        model, model_metadata = load_model(model_path)
        val_iter = iter_msd_patient_slices(cases, split["validation"], metadata,
            resolution=resolution, max_slices_per_patient=args.slices_per_patient)
        calibrated = calibrate_tumour_threshold(model, val_iter)
        accepted = calibrated.validation_macro_dice >= calibrated.baseline_validation_macro_dice - 1e-6
        calibration = {
            "method": "validation-selected tumour probability threshold",
            "accepted": bool(accepted),
            "tumour_probability_threshold": calibrated.threshold if accepted else None,
            "validation_mean_macro_dice": calibrated.validation_macro_dice,
            "raw_argmax_validation_mean_macro_dice": calibrated.baseline_validation_macro_dice,
            "validation_slices": calibrated.validation_slices,
            "candidate_results": list(calibrated.candidates),
            "selection_rule": "maximise mean per-slice WT/TC/ET macro Dice; ties within 1e-6 prefer higher threshold",
            "test_labels_used_for_selection": False,
        }
        model._qmi_calibration = calibration if accepted else None
        model_metadata = dict(model_metadata or {})
        model_metadata["probability_calibration"] = calibration
        save_model(model, model_path, metadata=model_metadata)
        sidecar = model_path.with_suffix(".json")
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        sidecar_payload["probability_calibration"] = calibration
        _atomic_json(sidecar, sidecar_payload)

        kmeans = joblib.load(baseline_path)
        baselines = {"kmeans": kmeans, "watershed": WatershedMultiClassBaseline(kmeans)}
        val_case, val_summary = evaluate_methods(
            iter_msd_patient_slices(cases, split["validation"], metadata, resolution=resolution,
                                    max_slices_per_patient=args.slices_per_patient),
            model, baselines, resolution)
        test_case, test_summary = evaluate_methods(
            iter_msd_patient_slices(cases, split["test"], metadata, resolution=resolution,
                                    max_slices_per_patient=args.slices_per_patient),
            model, baselines, resolution)

        # Preserve pre-calibration reports for provenance; publish calibrated reports separately.
        for old_name in (f"msd_validation_per_slice_{resolution}.csv", f"msd_validation_summary_{resolution}.csv",
                         f"msd_test_per_slice_{resolution}.csv", f"msd_test_summary_{resolution}.csv",
                         f"msd_test_per_patient_{resolution}.csv"):
            old = reports / old_name
            if old.exists():
                backup = old.with_name(old.stem + "_uncalibrated" + old.suffix)
                if not backup.exists():
                    old.replace(backup)
        val_case.to_csv(reports / f"msd_validation_calibrated_per_slice_{resolution}.csv", index=False)
        val_summary.to_csv(reports / f"msd_validation_calibrated_summary_{resolution}.csv", index=False)
        test_case.to_csv(reports / f"msd_test_calibrated_per_slice_{resolution}.csv", index=False)
        test_summary.to_csv(reports / f"msd_test_calibrated_summary_{resolution}.csv", index=False)
        numeric_columns = [c for c in test_case.select_dtypes(include=["number"]).columns
                           if c not in ("resolution", "slice_index")]
        patient_summary = (test_case.groupby(["patient_id", "method"], dropna=False)[numeric_columns]
                           .mean(numeric_only=True).reset_index())
        patient_summary.to_csv(reports / f"msd_test_calibrated_per_patient_{resolution}.csv", index=False)
        summary["resolutions"][str(resolution)] = {
            "threshold": calibrated.threshold,
            "validation_macro_dice_at_threshold": calibrated.validation_macro_dice,
            "validation_slices_for_calibration": calibrated.validation_slices,
            "heldout_test_summary": test_summary.to_dict(orient="records"),
            "model_path": str(model_path),
        }
        chosen_text = f"threshold={calibrated.threshold:.3f}" if accepted else "raw argmax retained (calibration did not improve validation)"
        print(f"\n[{resolution}] {chosen_text}; "
              f"calibrated validation Dice={calibrated.validation_macro_dice:.4f}; "
              f"raw validation Dice={calibrated.baseline_validation_macro_dice:.4f}; "
              f"validation slices={calibrated.validation_slices}")
        print(test_summary.to_string(index=False))

    out = reports / "msd_probability_calibration_summary.json"
    _atomic_json(out, summary)
    print(f"\nCalibration complete. Validation-selected models and held-out reports saved.\n{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
