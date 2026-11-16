from __future__ import annotations
import importlib
import platform
import shutil
import sys

REQUIRED = ["numpy", "scipy", "pandas", "sklearn", "skimage", "cv2", "nibabel", "matplotlib", "plotly", "streamlit", "fastapi", "uvicorn", "qiskit", "qiskit_aer"]
OPTIONAL = ["torch"]

def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.architecture()[0]})")
    failures = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            print(f"[OK] {name}: {getattr(module, '__version__', 'imported')}")
        except Exception as exc:
            print(f"[MISSING/ERROR] {name}: {type(exc).__name__}: {exc}")
            failures.append(name)
    for name in OPTIONAL:
        try:
            module = importlib.import_module(name)
            print(f"[OPTIONAL] {name}: {getattr(module, '__version__', 'imported')}")
            if name == "torch":
                print(f"         CUDA available: {bool(module.cuda.is_available())}")
        except Exception:
            print(f"[OPTIONAL] {name}: not installed (not required by the Random Forest pipeline)")
    nvidia_smi = shutil.which("nvidia-smi")
    print(f"NVIDIA driver utility: {'available' if nvidia_smi else 'not found on PATH'}")
    try:
        from qiskit_aer import AerSimulator
        print(f"Aer devices: {AerSimulator().available_devices()}")
    except Exception as exc:
        print(f"Aer device probe failed: {exc}")
        failures.append("qiskit-aer-device-probe")
    if failures:
        print("\nEnvironment check failed:", ", ".join(failures))
        return 1
    print("\nEnvironment check passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
