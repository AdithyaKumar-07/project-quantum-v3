from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmi.data.msd_dataset import read_msd_metadata, discover_msd_cases, load_msd_case, case_to_slices
from qmi.data.preprocessing import resize_slice
from qmi.models import load_model, predict_slice
from qmi.hybrid import run_hybrid_path
from qmi.quantum.simulation import profile_circuit
from qmi.metrics import reconstruction_metrics, segmentation_metrics, json_safe
from qmi.quantum.representations import encode_frqi_state, encode_neqr_state, decode_frqi_state, decode_neqr_state


def _load_heldout_cases(cases, split_path: Path, explicit_case_id: str | None,
                        max_cases: int, allow_non_test_case: bool):
    by_id = {case.case_id: case for case in cases}
    split = None
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    if explicit_case_id:
        if explicit_case_id not in by_id:
            raise SystemExit(f"Case ID not found in labelled dataset: {explicit_case_id}")
        if split and explicit_case_id not in split.get("test", []) and not allow_non_test_case:
            raise SystemExit(
                f"{explicit_case_id} is not in the saved held-out test partition. "
                "Use a test patient or pass --allow-non-test-case for exploratory work."
            )
        selected = [by_id[explicit_case_id]]
    else:
        if not split:
            raise SystemExit("No saved patient split found. Train first using scripts/train_msd.py.")
        test_ids = split.get("test", [])
        if not test_ids:
            raise SystemExit("Saved split does not contain a test partition.")
        missing = [case_id for case_id in test_ids if case_id not in by_id]
        if missing:
            raise SystemExit(f"Saved test split references missing cases: {missing[:5]}")
        selected = [by_id[case_id] for case_id in test_ids]
    if max_cases > 0:
        selected = selected[:max_cases]
    if not selected:
        raise SystemExit("No cases selected for the quantum experiment.")
    return selected, split


def _representative_tumour_slice(modalities: np.ndarray, labels: np.ndarray):
    selected = case_to_slices(modalities, labels, max_slices=8, tumor_first=True, return_indices=True)
    positive = [(z, m, y) for z, m, y in selected if np.any(y > 0)]
    if not positive:
        raise RuntimeError("No tumour-positive slice could be selected for this case")
    return max(positive, key=lambda item: int(np.count_nonzero(item[2])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out MSD FRQI/NEQR reconstruction and hybrid segmentation study")
    parser.add_argument("--data", default=str(ROOT / "data/raw/Task01_BrainTumour"))
    parser.add_argument("--case-id", default=None, help="Optional explicit patient ID; must be in held-out test split by default")
    parser.add_argument("--max-cases", type=int, default=1, help="Number of held-out patients; 0 processes all held-out patients")
    parser.add_argument("--allow-non-test-case", action="store_true", help="Allow an explicitly selected case outside the held-out test partition")
    parser.add_argument("--slice-index", type=int, default=None, help="Optional explicit axial slice; should contain tumour")
    parser.add_argument("--resolutions", choices=["16", "32", "both"], default="both")
    parser.add_argument("--mode", choices=["ideal", "noisy", "both"], default="ideal")
    parser.add_argument("--shots", type=int, default=50000)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--one-qubit-error", type=float, default=0.001)
    parser.add_argument("--two-qubit-error", type=float, default=0.01)
    parser.add_argument("--readout-error", type=float, default=0.01)
    parser.add_argument("--allow-expensive-18-qubit", action="store_true",
                        help="Allow full Aer simulation of 32x32 NEQR (18 qubits), which may be costly on CPU")
    args = parser.parse_args()
    if args.max_cases < 0 or args.shots < 1:
        parser.error("--max-cases must be >=0 and --shots must be positive")
    if any(not 0 <= x < 1 for x in (args.one_qubit_error, args.two_qubit_error, args.readout_error)):
        parser.error("Noise probabilities must lie in [0,1)")

    metadata = read_msd_metadata(args.data)
    cases = discover_msd_cases(args.data)
    selected_cases, split = _load_heldout_cases(cases, ROOT / "outputs/reports/msd_patient_split.json",
                                                  args.case_id, args.max_cases, args.allow_non_test_case)
    resolutions = (16, 32) if args.resolutions == "both" else (int(args.resolutions),)
    modes = ("ideal", "noisy") if args.mode == "both" else (args.mode,)
    out_dir = ROOT / "outputs" / "reports" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    profile_cache: dict[tuple[int, str], dict] = {}
    for case in selected_cases:
        volumes, masks = load_msd_case(case, metadata)
        if args.slice_index is None:
            slice_idx, native_modalities, native_truth = _representative_tumour_slice(volumes, masks)
        else:
            slice_idx = args.slice_index
            if not 0 <= slice_idx < masks.shape[0]:
                raise SystemExit(f"{case.case_id}: slice-index must be in [0,{masks.shape[0]-1}]")
            native_modalities, native_truth = volumes[:, slice_idx], masks[slice_idx]
            if not np.any(native_truth > 0):
                raise SystemExit(f"{case.case_id} slice {slice_idx} has no tumour; select a positive slice")
        for resolution in resolutions:
            modalities, truth = resize_slice(native_modalities, native_truth, resolution)
            model_file = ROOT / "artifacts" / f"msd_rf_{resolution}.joblib"
            if not model_file.is_file():
                raise SystemExit(f"Trained model missing: {model_file}. Run scripts/train_msd.py first.")
            model, model_metadata = load_model(model_file)
            gray8 = np.rint(modalities[0] * 255).astype(np.uint8)
            baseline_start = time.perf_counter()
            original_prediction = predict_slice(model, modalities)
            baseline_seconds = time.perf_counter() - baseline_start
            baseline_record = {
                "case_id": case.case_id, "slice_index": int(slice_idx), "resolution": resolution,
                "representation": "ORIGINAL_FLAIR", "mode": "classical_baseline",
                "reconstruction": {"mse": 0.0, "psnr_db": float("inf"), "ssim": 1.0},
                "segmentation": segmentation_metrics(original_prediction, truth),
                "inference_seconds": baseline_seconds,
                "patient_partition": "test" if split and case.case_id in split.get("test", []) else "exploratory_non_test",
                "model_metadata": model_metadata,
            }
            records.append(baseline_record)
            np.savez_compressed(out_dir / f"{case.case_id}_{resolution}_original_baseline.npz",
                                original_flair=gray8, prediction=original_prediction, ground_truth=truth)

            for representation in ("FRQI", "NEQR"):
                state = encode_frqi_state(gray8) if representation == "FRQI" else encode_neqr_state(gray8)
                decoder = decode_frqi_state if representation == "FRQI" else decode_neqr_state
                exact_reconstruction = decoder(state, resolution)
                exact_scores = reconstruction_metrics(gray8, exact_reconstruction, data_range=255)
                profile_key = (resolution, representation)
                if profile_key not in profile_cache:
                    profile_cache[profile_key] = profile_circuit(gray8, representation, max_qubits=16)
                profile = profile_cache[profile_key]
                profile_path = out_dir / f"{representation.lower()}_{resolution}_resource_profile.json"
                profile_path.write_text(json.dumps(json_safe(profile), indent=2, allow_nan=False), encoding="utf-8")
                records.append({
                    "case_id": case.case_id, "slice_index": int(slice_idx), "resolution": resolution,
                    "representation": representation, "mode": "exact_numpy_statevector_roundtrip",
                    "reconstruction": exact_scores, "qubits": profile.get("n_qubits"),
                    "resource_profile": profile,
                    "patient_partition": baseline_record["patient_partition"],
                })
                for mode in modes:
                    if representation == "NEQR" and resolution == 32 and not args.allow_expensive_18_qubit:
                        records.append({
                            "case_id": case.case_id, "slice_index": int(slice_idx), "resolution": resolution,
                            "representation": representation, "mode": f"aer_{mode}_skipped",
                            "reason": "18-qubit NEQR Aer execution is opt-in due CPU transpilation/simulation cost",
                            "patient_partition": baseline_record["patient_partition"],
                        })
                        continue
                    start = time.perf_counter()
                    hybrid = run_hybrid_path(
                        modalities, model, representation, true_labels=truth,
                        simulation_mode=mode, shots=args.shots if mode == "noisy" else None,
                        device=args.device,
                        noise_parameters={"one_qubit_error": args.one_qubit_error,
                                          "two_qubit_error": args.two_qubit_error,
                                          "readout_error": args.readout_error},
                    )
                    elapsed = time.perf_counter() - start
                    filename = f"{case.case_id}_{representation.lower()}_{resolution}_{mode}"
                    np.savez_compressed(out_dir / f"{filename}.npz",
                        original_flair=gray8, reconstructed_flair=hybrid.reconstructed_flair,
                        original_prediction=original_prediction, prediction=hybrid.predicted_labels,
                        ground_truth=truth)
                    records.append({
                        "case_id": case.case_id, "slice_index": int(slice_idx), "resolution": resolution,
                        "representation": representation, "mode": mode,
                        "reconstruction": hybrid.reconstruction,
                        "segmentation": hybrid.segmentation,
                        "delta_vs_original_macro_dice_active": (
                            hybrid.segmentation["macro_dice_active"] - baseline_record["segmentation"]["macro_dice_active"]
                            if hybrid.segmentation and np.isfinite(hybrid.segmentation["macro_dice_active"])
                            and np.isfinite(baseline_record["segmentation"]["macro_dice_active"]) else None
                        ),
                        "simulator_metadata": hybrid.simulator_metadata,
                        "simulation_seconds": hybrid.simulation_seconds,
                        "inference_seconds": hybrid.inference_seconds,
                        "wall_seconds": elapsed,
                        "patient_partition": baseline_record["patient_partition"],
                        "model_metadata": model_metadata,
                    })
                    active_dice = hybrid.segmentation["macro_dice_active"] if hybrid.segmentation else float("nan")
                    print(f"{case.case_id} {resolution} {representation} {mode}: "
                          f"MSE={hybrid.reconstruction['mse']:.6f}; active Macro Dice="
                          f"{active_dice:.4f}; {elapsed:.1f}s")

    result = {
        "dataset": "Medical Segmentation Decathlon Task01_BrainTumour",
        "patient_split_file": str(ROOT / "outputs/reports/msd_patient_split.json"),
        "case_count": len(selected_cases), "case_ids": [case.case_id for case in selected_cases],
        "results": records,
        "warning": "Research-only simulator measurements; no clinical performance or quantum speedup is claimed.",
    }
    output_path = out_dir / "msd_quantum_study.json"
    output_path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(records)} records for {len(selected_cases)} held-out case(s) to {output_path}")

if __name__ == "__main__":
    main()
