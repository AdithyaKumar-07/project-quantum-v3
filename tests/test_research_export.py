import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_research_figures.py"
_SPEC = importlib.util.spec_from_file_location("export_research_figures", _SCRIPT)
exporter = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(exporter)


def test_exporter_creates_classical_and_quantum_figures_from_real_report_names(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    figures = tmp_path / "figures"
    reports.mkdir()
    monkeypatch.setattr(exporter, "REPORTS", reports)
    monkeypatch.setattr(exporter, "FIGURES", figures)

    pd.DataFrame([{
        "method": "RandomForest",
        "macro_dice_active_mean": 0.60,
        "macro_iou_active_mean": 0.45,
    }]).to_csv(reports / "msd_test_summary_16.csv", index=False)
    pd.DataFrame([
        {"method": "RandomForest", "macro_dice_active": 0.60},
        {"method": "Otsu", "macro_dice_active": 0.20},
    ]).to_csv(reports / "msd_test_per_slice_16.csv", index=False)

    quantum = reports / "quantum"
    quantum.mkdir()
    truth = np.zeros((16, 16), dtype=np.uint8)
    truth[4:8, 4:8] = 2
    np.savez_compressed(
        quantum / "CASE_001_frqi_16_ideal.npz",
        original_flair=np.zeros((16, 16), dtype=np.uint8),
        reconstructed_flair=np.ones((16, 16), dtype=np.uint8),
        ground_truth=truth,
        original_prediction=truth,
        prediction=truth,
    )
    (quantum / "msd_quantum_study.json").write_text(json.dumps({
        "results": [{"case_id": "CASE_001", "representation": "FRQI", "resolution": 16, "mode": "ideal"}]
    }), encoding="utf-8")

    created = exporter._classical_figures((16,)) + exporter._quantum_figures()
    assert len(created) == 3
    assert all(Path(item).is_file() for item in created)


def test_exporter_skips_missing_reports(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    monkeypatch.setattr(exporter, "REPORTS", reports)
    monkeypatch.setattr(exporter, "FIGURES", tmp_path / "figures")
    reports.mkdir()
    assert exporter._classical_figures((16, 32, 240)) == []
    assert exporter._quantum_figures() == []
