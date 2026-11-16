import numpy as np
import pytest
from qmi.data.msd import modality_index_map, label_index_map, remap_msd_labels
from qmi.data.msd_dataset import case_to_slices


def test_official_msd_modality_mapping():
    mapping = modality_index_map({"0": "FLAIR", "1": "T1", "2": "T1gd", "3": "T2"})
    assert mapping == {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}


def test_msd_label_mapping_and_wt_tc_et_semantics():
    mapping = label_index_map({"0": "background", "1": "edema", "2": "non-enhancing tumour", "3": "enhancing tumour"})
    source = np.array([0, 1, 2, 3], dtype=np.uint8)
    internal = remap_msd_labels(source, mapping)
    np.testing.assert_array_equal(internal, np.array([0, 2, 1, 4], dtype=np.uint8))
    assert np.isin(internal, [1, 2, 4]).tolist() == [False, True, True, True]
    assert np.isin(internal, [1, 4]).tolist() == [False, False, True, True]
    assert (internal == 4).tolist() == [False, False, False, True]


def test_unknown_msd_label_is_rejected():
    with pytest.raises(ValueError, match="absent from dataset.json"):
        remap_msd_labels(np.array([0, 1, 9]), {0: 0, 1: 2})


def test_native_slice_selector_keeps_modalities_and_mask_aligned():
    modalities = np.zeros((4, 6, 12, 12), dtype=np.float32)
    labels = np.zeros((6, 12, 12), dtype=np.uint8)
    labels[2, 3:5, 3:5] = 1
    labels[3, 4:6, 4:6] = 2
    labels[4, 5:7, 5:7] = 4
    selected = case_to_slices(modalities, labels, max_slices=4, return_indices=True)
    assert selected
    for z, image_slice, label_slice in selected:
        assert image_slice.shape == (4, 12, 12)
        assert label_slice.shape == (12, 12)
        assert 0 <= z < 6
    assert len(selected) <= 4


def test_discover_msd_cases_ignores_appledouble_sidecars(tmp_path):
    import json
    from qmi.data.msd_dataset import read_msd_metadata, discover_msd_cases
    root = tmp_path / "Task01_BrainTumour"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir(parents=True)
    (root / "dataset.json").write_text(json.dumps({
        "modality": {"0": "FLAIR", "1": "T1", "2": "T1gd", "3": "T2"},
        "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumour", "3": "enhancing tumour"}
    }), encoding="utf-8")
    for case_id in ("BRATS_001", "BRATS_002"):
        (root / "imagesTr" / f"{case_id}.nii.gz").write_bytes(b"test")
        (root / "labelsTr" / f"{case_id}.nii.gz").write_bytes(b"test")
    (root / "imagesTr" / "._BRATS_001.nii.gz").write_bytes(b"metadata")
    metadata = read_msd_metadata(root)
    cases = discover_msd_cases(root)
    assert metadata.modalities == {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}
    assert [case.case_id for case in cases] == ["BRATS_001", "BRATS_002"]
