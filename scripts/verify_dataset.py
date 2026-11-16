from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmi.data.msd_dataset import read_msd_metadata, discover_msd_cases

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MSD Task01_BrainTumour dataset layout and metadata")
    parser.add_argument("--data", default=str(ROOT / "data/raw/Task01_BrainTumour"))
    args = parser.parse_args()
    root = Path(args.data)
    meta = read_msd_metadata(root)
    cases = discover_msd_cases(root)
    print("Dataset: Medical Segmentation Decathlon Task01_BrainTumour")
    print(f"Root: {root.resolve()}")
    print(f"Labelled image/label pairs: {len(cases)}")
    print(f"Channel indices (canonical order FLAIR,T1,T1ce,T2): {meta.modalities}")
    print(f"Source -> internal label mapping: {meta.labels}")
    if len(cases) < 10:
        raise SystemExit("At least 10 labelled cases are needed for train/validation/test splitting.")
    print("Dataset structure and metadata validation passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
