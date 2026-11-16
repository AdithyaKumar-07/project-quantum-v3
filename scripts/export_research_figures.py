"""Export reproducible figures from real MSD and quantum report files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"


def _reported_path(path: Path) -> str:
    """Return project-relative paths, while allowing isolated test fixtures."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _classical_figures(resolutions: tuple[int, ...]) -> list[str]:
    created = []
    for resolution in resolutions:
        calibrated_summary = REPORTS / f"msd_test_calibrated_summary_{resolution}.csv"
        calibrated_slices = REPORTS / f"msd_test_calibrated_per_slice_{resolution}.csv"
        summary_path = calibrated_summary if calibrated_summary.is_file() else REPORTS / f"msd_test_summary_{resolution}.csv"
        slices_path = calibrated_slices if calibrated_slices.is_file() else REPORTS / f"msd_test_per_slice_{resolution}.csv"
        if not summary_path.is_file() or not slices_path.is_file():
            continue
        summary = pd.read_csv(summary_path)
        per_slice = pd.read_csv(slices_path)
        if "macro_dice_active_mean" not in summary.columns:
            raise ValueError(f"{summary_path} lacks active-region macro Dice. Re-run training with the current version.")
        FIGURES.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(summary))
        width = 0.36
        for offset, metric, label in [(-width / 2, "macro_dice_active_mean", "Active-region macro Dice"),
                                       (width / 2, "macro_iou_active_mean", "Active-region macro IoU")]:
            values = summary[metric].to_numpy(dtype=float)
            ax.bar(x + offset, values, width, label=label)
        ax.set_xticks(x, summary["method"].astype(str), rotation=12)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        calibration_note = " · validation-calibrated" if calibrated_summary.is_file() else " · uncalibrated"
        ax.set_title(f"Held-out MSD test comparison · {resolution}×{resolution if resolution != 240 else 240}{calibration_note}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        output = FIGURES / f"msd_test_method_comparison_{resolution}.png"
        fig.savefig(output, dpi=180)
        plt.close(fig)
        created.append(_reported_path(output))

        fig, ax = plt.subplots(figsize=(9, 5))
        methods = list(per_slice["method"].dropna().unique())
        data = [per_slice.loc[per_slice["method"] == method, "macro_dice_active"].dropna().to_numpy() for method in methods]
        data = [values for values in data if len(values)]
        labels = [method for method, values in zip(methods, data) if len(values)]
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Active-region macro Dice")
            ax.set_title(f"Held-out per-slice dispersion · {resolution}×{resolution if resolution != 240 else 240}")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            output = FIGURES / f"msd_test_slice_dispersion_{resolution}.png"
            fig.savefig(output, dpi=180)
            created.append(_reported_path(output))
        plt.close(fig)
    return created


def _quantum_figures() -> list[str]:
    result_path = REPORTS / "quantum" / "msd_quantum_study.json"
    if not result_path.is_file():
        return []
    data = json.loads(result_path.read_text(encoding="utf-8"))
    created = []
    for record in data.get("results", []):
        if record.get("mode") not in {"ideal", "noisy"}:
            continue
        case_id = record.get("case_id")
        rep = str(record.get("representation", "")).lower()
        resolution = record.get("resolution")
        mode = record.get("mode")
        path = REPORTS / "quantum" / f"{case_id}_{rep}_{resolution}_{mode}.npz"
        if not path.is_file():
            continue
        with np.load(path) as arrays:
            original = arrays["original_flair"]
            reconstructed = arrays["reconstructed_flair"]
            truth = arrays["ground_truth"]
            baseline = arrays["original_prediction"]
            hybrid = arrays["prediction"]
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
        panels = [(original, "Original FLAIR", "gray"), (reconstructed, "Reconstructed FLAIR", "gray"),
                  (truth, "Ground truth", "nipy_spectral"), (baseline, "Original FLAIR RF", "nipy_spectral"),
                  (hybrid, f"{record['representation']} hybrid RF", "nipy_spectral")]
        for ax, (image, title, cmap) in zip(axes, panels):
            ax.imshow(image, cmap=cmap, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{case_id} · {rep.upper()} · {resolution}×{resolution} · {mode}")
        fig.tight_layout()
        output = FIGURES / f"{case_id}_{rep}_{resolution}_{mode}_comparison.png"
        fig.savefig(output, dpi=180)
        plt.close(fig)
        created.append(_reported_path(output))
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Export figures from real MSD evaluation and quantum experiment reports")
    parser.add_argument("--resolutions", choices=["16", "32", "240", "both", "all"], default="all")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from qmi.experiments.resolutions import parse_resolutions
    resolutions = parse_resolutions(args.resolutions)
    created = _classical_figures(resolutions) + _quantum_figures()
    manifest = {"figures": created, "count": len(created),
                "source_note": "All figures are generated from current outputs/reports. Synthetic demo reports are not read."}
    FIGURES.mkdir(parents=True, exist_ok=True)
    (REPORTS / "research_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not created:
        print("No real-data report files were found. Run training and/or quantum experiments first.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
