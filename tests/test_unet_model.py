import numpy as np
import pytest

torch = pytest.importorskip("torch")
from qmi.unet import (UNet2D, encode_internal_labels, decode_class_indices,
                      combined_segmentation_loss, select_patient_slice_indices)


def test_internal_label_mapping_round_trip():
    labels = np.array([[0, 1, 2, 4]], dtype=np.uint8)
    encoded = encode_internal_labels(labels)
    assert encoded.tolist() == [[0, 1, 2, 3]]
    assert np.array_equal(decode_class_indices(encoded), labels)


def test_slice_selector_keeps_positive_and_background_examples():
    labels = np.zeros((20, 32, 32), dtype=np.uint8)
    labels[5, 10:15, 10:15] = 1
    labels[7, 12:16, 12:16] = 2
    labels[9, 14:18, 14:18] = 4
    selected = select_patient_slice_indices(labels, max_slices=8)
    assert len(selected) <= 8
    assert set((5, 7, 9)).issubset(set(selected.tolist()))
    assert any(not np.any(labels[z]) for z in selected)


def test_unet_forward_shape_and_loss_are_finite():
    model = UNet2D(base_channels=8, dropout=0.0)
    images = torch.rand(2, 4, 64, 64)
    targets = torch.randint(0, 4, (2, 64, 64))
    logits = model(images)
    assert tuple(logits.shape) == (2, 4, 64, 64)
    loss = combined_segmentation_loss(logits, targets, torch.ones(4))
    assert torch.isfinite(loss)
    loss.backward()


def test_unet_rejects_wrong_input_channels():
    model = UNet2D(base_channels=8)
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 64, 64))
