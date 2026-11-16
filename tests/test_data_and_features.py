import numpy as np
from qmi.data.synthetic import make_synthetic_slice, make_synthetic_dataset
from qmi.data.preprocessing import resize_slice
from qmi.features import extract_features
from qmi.data.splits import split_patients


def test_synthetic_slice_shapes_and_labels():
    s = make_synthetic_slice(seed=4, size=64)
    assert s.modalities.shape == (4, 64, 64)
    assert s.label.shape == (64, 64)
    assert set(np.unique(s.label)).issubset({0, 1, 2, 4})


def test_resize_keeps_label_values_and_dimensions():
    s = make_synthetic_slice(seed=5, size=64)
    x, y = resize_slice(s.modalities, s.label, 32)
    assert x.shape == (4, 32, 32)
    assert y.shape == (32, 32)
    assert set(np.unique(y)).issubset({0, 1, 2, 4})


def test_features_are_finite_and_expected_width():
    s = make_synthetic_slice(seed=6, size=32)
    f = extract_features(s.modalities)
    assert f.shape == (32*32, 12)
    assert np.isfinite(f).all()


def test_patient_split_is_disjoint_and_complete():
    ids = [f"p{i}" for i in range(20)]
    split = split_patients(ids, seed=7)
    sets = [set(split[k]) for k in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(ids)


def test_case_to_slices_can_preserve_original_slice_indices():
    from qmi.data.brats import case_to_slices

    modalities = np.zeros((4, 5, 8, 8), dtype=np.float32)
    labels = np.zeros((5, 8, 8), dtype=np.uint8)
    labels[2, 2:4, 2:4] = 4
    selected = case_to_slices(modalities, labels, max_slices=2, return_indices=True)
    assert selected
    assert all(len(item) == 3 for item in selected)
    assert any(item[0] == 2 for item in selected)
