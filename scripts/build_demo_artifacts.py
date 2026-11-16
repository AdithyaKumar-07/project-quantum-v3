from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sklearn
import numpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmi.data.synthetic import make_synthetic_dataset
from qmi.data.preprocessing import resize_slice
from qmi.experiments.pipeline import split_ids, fit_classical_baselines, evaluate_methods
from qmi.models import balanced_pixel_samples, train_random_forest, save_model, predict_slice


def main():
    samples = make_synthetic_dataset(n_samples=36, size=64, seed=2026)
    data_path = ROOT / "data/demo/synthetic_mri_demo.npz"
    np.savez_compressed(data_path,
        modalities=np.stack([s.modalities for s in samples]),
        labels=np.stack([s.label for s in samples]),
        sample_ids=np.array([s.sample_id for s in samples]))
    splits = split_ids([s.sample_id for s in samples], seed=42)
    by_id = {s.sample_id: s for s in samples}
    train = [by_id[k] for k in splits["train"]]
    test = [by_id[k] for k in splits["test"]]
    test_results = []
    reports = {}
    models = {}
    baselines_by_resolution = {}
    for resolution in (16, 32):
        train_resized = [(s.sample_id, *resize_slice(s.modalities, s.label, resolution)) for s in train]
        test_resized = [(s.sample_id, *resize_slice(s.modalities, s.label, resolution)) for s in test]
        X, y = balanced_pixel_samples([(m, lab) for _,m,lab in train_resized], samples_per_class=400, random_state=42)
        model, report = train_random_forest(X, y, n_estimators=80, max_depth=16, random_state=42, n_jobs=-1)
        model_path = ROOT / f"artifacts/demo_rf_{resolution}.joblib"
        save_model(model, model_path, metadata={
            "dataset": "SYNTHETIC_DEMO_ONLY", "not_for_research": True,
            "resolution": resolution, "training_pixels": report.n_samples,
            "class_counts": report.class_counts, "random_state": 42})
        kmeans_slices = [(m, lab) for _,m,lab in train_resized]
        baselines = fit_classical_baselines(kmeans_slices)
        joblib.dump(baselines["kmeans"], ROOT / f"artifacts/demo_kmeans_{resolution}.joblib", compress=3)
        per_case, summary = evaluate_methods(test_resized, model, baselines, resolution=resolution)
        per_case.to_csv(ROOT / f"outputs/reports/demo_per_case_metrics_{resolution}.csv", index=False)
        summary.to_csv(ROOT / f"outputs/reports/demo_summary_metrics_{resolution}.csv", index=False)
        test_results.append(per_case)
        reports[str(resolution)] = report.__dict__
        models[str(resolution)] = str(model_path.relative_to(ROOT))
        baselines_by_resolution[str(resolution)] = str((ROOT / f"artifacts/demo_kmeans_{resolution}.joblib").relative_to(ROOT))
    pd = __import__("pandas")
    combined = pd.concat(test_results, ignore_index=True)
    combined.to_csv(ROOT / "outputs/reports/demo_all_resolutions_per_case.csv", index=False)
    numeric_cols = [c for c in combined.columns if c not in ("sample_id", "method", "resolution")]
    aggregate = combined.groupby(["method", "resolution"])[numeric_cols].agg(["mean", "std"])
    aggregate.columns = [f"{a}_{b}" for a,b in aggregate.columns]
    aggregate.reset_index().to_csv(ROOT / "outputs/reports/demo_all_resolutions_summary.csv", index=False)
    # Generate demo figure at 32x32.
    sample = test[0]
    m, y = resize_slice(sample.modalities, sample.label, 32)
    from qmi.models import load_model
    model, _ = load_model(ROOT / "artifacts/demo_rf_32.joblib")
    prediction = predict_slice(model, m)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(m[0], cmap="gray"); axes[0].set_title("FLAIR")
    axes[1].imshow(y, vmin=0, vmax=4, cmap="nipy_spectral"); axes[1].set_title("Synthetic ground truth")
    axes[2].imshow(prediction, vmin=0, vmax=4, cmap="nipy_spectral"); axes[2].set_title("RF prediction")
    axes[3].imshow(m[0], cmap="gray"); axes[3].imshow(prediction, alpha=.4, vmin=0, vmax=4, cmap="nipy_spectral"); axes[3].set_title("Overlay")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/demo_prediction.png", dpi=140); plt.close(fig)
    meta = {"status": "SMOKE_TEST_ONLY", "dataset": "synthetic MRI-like data",
            "warning": "These results are not BraTS results, not clinical evidence, and must not be reported as medical research findings.",
            "n_samples": len(samples), "train_samples": len(train), "test_samples": len(test),
            "split_ids": splits, "models": models, "kmeans_baselines": baselines_by_resolution,
            "training_reports": reports,
            "environment": {"python_version": __import__("sys").version,
                            "numpy_version": numpy.__version__,
                            "scikit_learn_version": sklearn.__version__}}
    (ROOT / "artifacts/demo_training_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Demo data: {data_path}")
    print(f"Demo model artifacts: {models}")
    print(f"Training reports: {json.dumps(reports)}")
    print(combined.groupby(["method", "resolution"])[["macro_dice", "macro_iou", "WT_dice", "TC_dice", "ET_dice"]].mean().to_string())

if __name__ == "__main__":
    main()
