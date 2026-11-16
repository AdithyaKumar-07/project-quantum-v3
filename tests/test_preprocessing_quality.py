import numpy as np
import pytest
from qmi.data.preprocessing import resize_slice
from qmi.metrics import segmentation_metrics, json_safe


def test_occupancy_downsample_preserves_tiny_tumour_labels():
    modalities = np.zeros((4, 240, 240), dtype=np.float32)
    labels = np.zeros((240, 240), dtype=np.uint8)
    labels[100, 100] = 4
    m, y = resize_slice(modalities, labels, 32, label_mode="occupancy")
    assert m.shape == (4, 32, 32)
    assert y.shape == (32, 32)
    assert np.count_nonzero(y == 4) >= 1


def test_nearest_downsample_is_available_for_legacy_reproduction():
    modalities = np.zeros((4, 240, 240), dtype=np.float32)
    labels = np.zeros((240, 240), dtype=np.uint8)
    labels[100, 100] = 4
    _, y = resize_slice(modalities, labels, 32, label_mode="nearest")
    assert y.shape == (32, 32)


def test_native_resolution_does_not_resize_or_mutate_labels():
    modalities = np.random.default_rng(5).random((4, 240, 240), dtype=np.float32)
    labels = np.zeros((240, 240), dtype=np.uint8)
    labels[20:24, 20:24] = 4
    m, y = resize_slice(modalities, labels, 240)
    np.testing.assert_array_equal(m, modalities)
    np.testing.assert_array_equal(y, labels)


def test_native_mode_rejects_non_native_shape():
    with pytest.raises(ValueError, match="Native mode expects"):
        resize_slice(np.zeros((4, 128, 128), dtype=np.float32), np.zeros((128, 128), dtype=np.uint8), 240)


def test_present_gt_macro_does_not_count_empty_empty_as_perfect():
    truth = np.zeros((16, 16), dtype=np.uint8)
    truth[5, 5] = 2
    pred = np.zeros_like(truth)
    scores = segmentation_metrics(pred, truth)
    assert scores["WT"]["dice"] == 0.0
    assert scores["macro_dice"] > scores["macro_dice_active"]
    assert scores["macro_dice_active"] == 0.0
    assert scores["n_regions_present_gt"] == 1


def test_json_safe_converts_nonfinite_and_numpy_values():
    import math
    converted = json_safe({"value": np.float64(np.nan), "items": [np.int64(3), float("inf")]})
    assert converted == {"value": None, "items": [3, None]}
