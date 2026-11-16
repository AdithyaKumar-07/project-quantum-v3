from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmi.experiments.runner import run_msd_experiment
from qmi.experiments.resolutions import parse_resolutions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate MSD brain-tumour segmentation at 16x16, 32x32 and native 240x240 resolution")
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "Task01_BrainTumour"))
    parser.add_argument("--max-patients", type=int, default=120,
                        help="Patient cap for the first full run; use 0 to use all labelled patients")
    parser.add_argument("--slices-per-patient", type=int, default=8)
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--trees", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", choices=["16", "32", "both", "native", "all"], default="both")
    args = parser.parse_args()
    if args.slices_per_patient < 1 or args.samples_per_class < 1 or args.trees < 10:
        parser.error("slices-per-patient and samples-per-class must be positive; trees must be >=10")
    resolutions = parse_resolutions(args.resolution)
    result = run_msd_experiment(args.data, max_patients=args.max_patients,
        max_slices_per_patient=args.slices_per_patient, samples_per_class=args.samples_per_class,
        trees=args.trees, seed=args.seed, resolutions=resolutions, output_root=ROOT)
    print("\nExperiment finished. Machine-readable summary:")
    print(json.dumps({"dataset": result["dataset"], "patients": result["selected_patients"],
                      "resolutions": list(result["resolutions"])}, indent=2))

if __name__ == "__main__":
    main()
