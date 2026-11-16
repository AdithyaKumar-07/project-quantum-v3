from __future__ import annotations
"""One-command best-model calibration, research figure refresh and release checks."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], label: str) -> None:
    print(f"\n===== {label} =====", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate current MSD models and finalize reports in one command")
    parser.add_argument("--resolutions", choices=["16", "32", "240", "both", "all"], default="all")
    parser.add_argument("--slices-per-patient", type=int, default=8)
    args = parser.parse_args()
    run([sys.executable, "scripts/optimize_msd_models.py", "--resolutions", args.resolutions,
         "--slices-per-patient", str(args.slices_per_patient)], "VALIDATION-ONLY MODEL CALIBRATION + HELD-OUT TEST EVALUATION")
    figure_resolution = args.resolutions
    if figure_resolution == "native":
        figure_resolution = "240"
    run([sys.executable, "scripts/export_research_figures.py", "--resolutions", figure_resolution],
        "RESEARCH FIGURE EXPORT")
    run([sys.executable, "scripts/validate_release.py", "--require-dataset", "--require-models", "--require-reports"],
        "RELEASE PREFLIGHT + AUTOMATED TEST SUITE")
    print("\nBEST-MODEL FINALIZATION COMPLETE. Thresholds were selected on validation patients; test patients were used only for final reporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
