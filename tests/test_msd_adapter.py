import numpy as np
import pytest
from qmi.data.msd import modality_index_map, label_index_map, remap_msd_labels


def test_modality_index_map_uses_metadata_names():
    actual = modality_index_map({"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"})
    assert actual == {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}


def test_msd_labels_convert_to_internal_brats_convention():
    mapping = label_index_map({"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumor"})
    assert mapping == {0: 0, 1: 2, 2: 1, 3: 4}
    source = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    assert remap_msd_labels(source, mapping).tolist() == [[0, 2, 1, 4]]


def test_unknown_label_metadata_fails_closed():
    with pytest.raises(ValueError):
        label_index_map({"0": "background", "1": "edema", "2": "tumor", "3": "enhancing tumor"})


def test_unknown_voxel_label_fails_closed():
    mapping = {0: 0, 1: 2, 2: 1, 3: 4}
    with pytest.raises(ValueError):
        remap_msd_labels(np.array([0, 1, 2, 9]), mapping)
