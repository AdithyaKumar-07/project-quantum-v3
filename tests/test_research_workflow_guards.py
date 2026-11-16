import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qmi.data.msd_dataset import MSDCase
from qmi.data import preprocessing
from qmi import training


def _load_quantum_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_quantum_study.py"
    spec = importlib.util.spec_from_file_location("run_quantum_study_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quantum_case_selection_requires_saved_heldout_split_by_default(tmp_path):
    module = _load_quantum_script()
    cases = [SimpleNamespace(case_id="CASE_A"), SimpleNamespace(case_id="CASE_B")]
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": ["CASE_A"], "validation": [], "test": ["CASE_B"]}))

    selected, split = module._load_heldout_cases(cases, split_path, None, 0, False)
    assert [case.case_id for case in selected] == ["CASE_B"]
    assert split["test"] == ["CASE_B"]

    with pytest.raises(SystemExit, match="not in the saved held-out test partition"):
        module._load_heldout_cases(cases, split_path, "CASE_A", 1, False)

    selected, _ = module._load_heldout_cases(cases, split_path, "CASE_A", 1, True)
    assert [case.case_id for case in selected] == ["CASE_A"]


def test_quantum_case_selection_requires_split_for_implicit_case_selection(tmp_path):
    module = _load_quantum_script()
    cases = [SimpleNamespace(case_id="CASE_A")]
    with pytest.raises(SystemExit, match="No saved patient split found"):
        module._load_heldout_cases(cases, tmp_path / "missing.json", None, 1, False)


def test_native_slice_iterator_yields_copied_native_arrays_and_tumour_slice(monkeypatch):
    case = MSDCase("CASE_A", Path("image.nii.gz"), Path("label.nii.gz"))
    modalities = np.zeros((4, 3, 240, 240), dtype=np.float32)
    labels = np.zeros((3, 240, 240), dtype=np.uint8)
    labels[1, 100:105, 100:105] = 2
    modalities[:, 1, 100:105, 100:105] = 0.8

    monkeypatch.setattr(training, "load_msd_case", lambda _case, _metadata: (modalities, labels))
    result = list(training.iter_msd_patient_slices([case], ["CASE_A"], object(), resolution=240, max_slices_per_patient=1))
    assert len(result) == 1
    sample_id, image_slice, label_slice = result[0]
    assert sample_id == "CASE_A_z001"
    assert image_slice.shape == (4, 240, 240)
    assert label_slice.shape == (240, 240)
    assert np.any(label_slice > 0)
    assert not np.shares_memory(image_slice, modalities)
    assert not np.shares_memory(label_slice, labels)
