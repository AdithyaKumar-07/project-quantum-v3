"""Release preflight for the Project Quantum V3 research bundle."""
from __future__ import annotations
import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = [
    "numpy", "scipy", "pandas", "sklearn", "skimage", "cv2", "nibabel",
    "matplotlib", "plotly", "streamlit", "fastapi", "uvicorn", "qiskit", "qiskit_aer",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Project Quantum V3 clean-release checks")
    parser.add_argument("--require-dataset", action="store_true")
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-reports", action="store_true")
    parser.add_argument("--require-unet", action="store_true", help="Require a loadable trained native-resolution U-Net checkpoint")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    print(f"Project root: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        failures.append("Project release environment is qualified for Python 3.12.x only")

    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            print(f"[OK] {module_name}: {getattr(module, '__version__', 'imported')}")
        except Exception as exc:
            print(f"[FAIL] {module_name}: {type(exc).__name__}: {exc}")
            failures.append(f"missing/broken dependency: {module_name}")

    try:
        import nbformat
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            print(f"[OK] notebook schema: {path.name} ({len(notebook.cells)} cells)")
    except Exception as exc:
        failures.append(f"notebook validation failed: {type(exc).__name__}: {exc}")

    if args.require_dataset:
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from qmi.data.msd_dataset import read_msd_metadata, discover_msd_cases
            metadata = read_msd_metadata(ROOT / "data/raw/Task01_BrainTumour")
            cases = discover_msd_cases(ROOT / "data/raw/Task01_BrainTumour")
            print(f"[OK] dataset pairs: {len(cases)}; modalities={metadata.modalities}; labels={metadata.labels}")
            if len(cases) != 484:
                failures.append(f"expected 484 labelled pairs in this verified dataset, found {len(cases)}")
        except Exception as exc:
            print(f"[FAIL] dataset: {type(exc).__name__}: {exc}")
            failures.append("dataset preflight failed")

    if args.require_models:
        missing = [f"artifacts/msd_rf_{r}.joblib" for r in (16, 32, 240)
                   if not (ROOT / f"artifacts/msd_rf_{r}.joblib").is_file()]
        if missing:
            failures.extend([f"missing trained model: {item}" for item in missing])
        else:
            print("[OK] real-data model artifacts found at 16, 32 and native 240")

    if args.require_unet:
        unet_path = ROOT / "artifacts" / "msd_unet_240.pt"
        meta_path = ROOT / "artifacts" / "msd_unet_240.json"
        try:
            import torch
            if not unet_path.is_file() or not meta_path.is_file():
                failures.append("missing U-Net checkpoint/metadata: artifacts/msd_unet_240.pt and .json")
            else:
                checkpoint = torch.load(unet_path, map_location="cpu", weights_only=True)
                if "state_dict" not in checkpoint or "model_config" not in checkpoint:
                    failures.append("U-Net checkpoint schema is invalid")
                else:
                    print(f"[OK] trained U-Net checkpoint: {unet_path.name} (epoch {checkpoint.get('metadata', {}).get('best_epoch', 'unknown')})")
        except Exception as exc:
            print(f"[FAIL] U-Net checkpoint: {type(exc).__name__}: {exc}")
            failures.append("U-Net checkpoint validation failed")

    if args.require_reports:
        required = [ROOT / "outputs/reports/msd_patient_split.json"]
        for r in (16, 32, 240):
            summary_cal = ROOT / f"outputs/reports/msd_test_calibrated_summary_{r}.csv"
            patient_cal = ROOT / f"outputs/reports/msd_test_calibrated_per_patient_{r}.csv"
            required.append(summary_cal if summary_cal.is_file() else ROOT / f"outputs/reports/msd_test_summary_{r}.csv")
            required.append(patient_cal if patient_cal.is_file() else ROOT / f"outputs/reports/msd_test_per_patient_{r}.csv")
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            failures.extend([f"missing final report: {item}" for item in missing])
        else:
            print("[OK] required classical held-out reports found")
        quantum = ROOT / "outputs/reports/quantum/msd_quantum_study.json"
        if not quantum.is_file():
            failures.append("missing quantum study report: outputs/reports/quantum/msd_quantum_study.json")

    if not args.skip_tests:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT)
        if result.returncode != 0:
            failures.append("pytest suite failed")
        else:
            print("[OK] pytest suite")

    if failures:
        print("\nRELEASE CHECK FAILED:")
        for item in failures:
            print(f" - {item}")
        return 1
    print("\nRelease preflight passed. Manual visual/research-paper checks are still required.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
