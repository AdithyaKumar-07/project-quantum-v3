from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmi.data.msd_dataset import read_msd_metadata, discover_msd_cases, load_msd_case
from qmi.models import load_model
from qmi.quantum.patchwise import run_patchwise_hybrid
from qmi.metrics import json_safe


def main() -> int:
    parser = argparse.ArgumentParser(description="High-resolution patch-wise FRQI/NEQR and native RF hybrid experiment")
    parser.add_argument("--data", default=str(ROOT / "data/raw/Task01_BrainTumour"))
    parser.add_argument("--case-id", required=True, help="Patient ID from the saved held-out test split")
    parser.add_argument("--slice-index", type=int, default=None, help="Tumour-positive axial slice; default selects the slice with the largest labelled area")
    parser.add_argument("--resolution", type=int, choices=[64, 128, 240], default=240)
    parser.add_argument("--representation", choices=["FRQI", "NEQR"], default="FRQI")
    parser.add_argument("--mode", choices=["statevector", "aer_ideal_full", "aer_ideal_spotcheck", "aer_noisy_spotcheck"], default="statevector")
    parser.add_argument("--max-aer-tiles", type=int, default=1)
    parser.add_argument("--shots", type=int, default=50000)
    parser.add_argument("--allow-expensive-18-qubit", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/reports/quantum/highres"))
    args = parser.parse_args()
    if not 1 <= args.max_aer_tiles <= 4:
        parser.error("--max-aer-tiles must be between 1 and 4")

    split_path = ROOT / "outputs/reports/msd_patient_split.json"
    if not split_path.is_file():
        raise SystemExit(f"Patient split not found: {split_path}. Run full-cohort training first.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if args.case_id not in split.get("test", []):
        raise SystemExit(f"{args.case_id} is not in the saved held-out test partition. No test-set override is provided by this script.")
    model_path = ROOT / "artifacts/msd_rf_240.joblib"
    if not model_path.is_file():
        raise SystemExit(f"Native 240x240 model not found: {model_path}")
    model, model_metadata = load_model(model_path)

    metadata = read_msd_metadata(args.data)
    case = next((c for c in discover_msd_cases(args.data) if c.case_id == args.case_id), None)
    if case is None:
        raise SystemExit(f"Held-out case {args.case_id} is not found under {args.data}")
    volumes, masks = load_msd_case(case, metadata)
    if args.slice_index is None:
        tumour_slices = np.flatnonzero(np.any(masks > 0, axis=(1, 2)))
        if len(tumour_slices) == 0:
            raise SystemExit(f"{args.case_id} has no labelled tumour slices")
        area = np.count_nonzero(masks[tumour_slices], axis=(1, 2))
        slice_index = int(tumour_slices[int(np.argmax(area))])
    else:
        slice_index = int(args.slice_index)
    if not 0 <= slice_index < masks.shape[0]:
        raise SystemExit(f"slice-index must be in [0,{masks.shape[0]-1}]")
    truth = masks[slice_index]
    if not np.any(truth > 0):
        raise SystemExit(f"{args.case_id} slice {slice_index} has no tumour. Select a tumour-positive slice.")

    result = run_patchwise_hybrid(
        volumes[:, slice_index], model, args.representation,
        true_labels=truth, quantum_resolution=args.resolution,
        mode=args.mode, max_aer_tiles=args.max_aer_tiles, shots=args.shots,
        allow_expensive_18_qubit=args.allow_expensive_18_qubit,
    )
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.case_id}_z{slice_index}_{args.resolution}_{args.representation.lower()}_{args.mode}"
    npz_path = out / f"{stem}.npz"
    np.savez_compressed(npz_path,
        native_original_flair=np.rint(volumes[0, slice_index] * 255).clip(0,255).astype(np.uint8),
        quantum_resolution_reconstruction=result.reconstructed_flair,
        native_reconstructed_flair=result.reconstructed_flair_native,
        prediction_native=result.predicted_labels_native,
        ground_truth_native=truth,
    )
    report = {
        "dataset": "Medical Segmentation Decathlon Task01_BrainTumour",
        "case_id": args.case_id, "slice_index": slice_index,
        "patient_partition": "test", "model_artifact": str(model_path),
        "model_metadata": model_metadata,
        "representation": result.representation,
        "reconstruction": result.reconstruction,
        "segmentation_native": result.segmentation,
        "metadata": result.metadata,
        "patch_records": result.patch_records,
        "aer_tile_results": result.aer_tile_records,
        "total_seconds": result.total_seconds,
        "npz_output": str(npz_path),
    }
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(json_path), "npz": str(npz_path),
        "reconstruction": result.reconstruction,
        "segmentation_active_macro_dice": (result.segmentation or {}).get("macro_dice_active"),
        "patches": result.metadata["total_patches"],
        "aer_tile_results": len(result.aer_tile_records),
        "total_seconds": result.total_seconds}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
