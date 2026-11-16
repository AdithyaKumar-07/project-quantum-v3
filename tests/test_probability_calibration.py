import numpy as np
from qmi.calibration import probabilities_to_labels, slice_macro_dice, calibrate_tumour_threshold


class ToyProbabilityModel:
    classes_ = np.array([0, 1, 2, 4], dtype=np.uint8)
    def predict_proba(self, features):
        # Feature column zero is the synthetic tumour confidence.
        confidence = np.clip(features[:, 0], 0, 1)
        return np.column_stack([1-confidence, confidence*0.2, confidence*0.3, confidence*0.5])


def test_probability_gate_suppresses_low_confidence_tumour_predictions():
    p = np.array([[0.95, 0.01, 0.02, 0.02], [0.20, 0.10, 0.20, 0.50]])
    labels = probabilities_to_labels(p, np.array([0, 1, 2, 4]), 0.5)
    np.testing.assert_array_equal(labels, np.array([0, 4], dtype=np.uint8))


def test_slice_macro_dice_rewards_correct_background_and_lesion_overlap():
    truth = np.zeros((8, 8), dtype=np.uint8)
    truth[3:5, 3:5] = 4
    assert slice_macro_dice(truth, truth) == 1.0
    assert slice_macro_dice(np.zeros_like(truth), truth) < 1.0


def test_threshold_calibration_uses_validation_labels_and_returns_candidate_trace():
    # Tiny feature-compatible image; first modality carries a synthetic confidence map.
    from qmi.features import extract_features
    class Model:
        classes_ = np.array([0, 1, 2, 4], dtype=np.uint8)
        def predict_proba(self, features):
            conf = np.clip(features[:, 0], 0, 1)
            return np.column_stack([1-conf, conf*0.2, conf*0.3, conf*0.5])
    modalities = np.zeros((4, 8, 8), dtype=np.float32)
    modalities[0, 2:6, 2:6] = 0.9
    truth = np.zeros((8, 8), dtype=np.uint8)
    truth[2:6, 2:6] = 4
    result = calibrate_tumour_threshold(Model(), [("val_case_z1", modalities, truth)], thresholds=[0.1, 0.5, 0.8, 0.95])
    assert result.validation_slices == 1
    assert result.threshold in {0.1, 0.5, 0.8, 0.95}
    assert 0 <= result.baseline_validation_macro_dice <= 1
    assert len(result.candidates) == 4
    assert 0 <= result.validation_macro_dice <= 1


def test_predict_slice_applies_embedded_calibration_threshold():
    from qmi.models import predict_slice
    class Model:
        classes_ = np.array([0, 1, 2, 4], dtype=np.uint8)
        _qmi_calibration = {"tumour_probability_threshold": 0.5}
        def predict(self, features):
            raise AssertionError("Calibrated inference must use predict_proba")
        def predict_proba(self, features):
            # Low confidence everywhere except one compact high-confidence lesion.
            confidence = np.full(features.shape[0], 0.1, dtype=np.float32)
            confidence[10:14] = 0.9
            return np.column_stack([1-confidence, confidence*0.1, confidence*0.2, confidence*0.7])
    modalities = np.zeros((4, 4, 4), dtype=np.float32)
    pred = predict_slice(Model(), modalities)
    assert pred.shape == (4, 4)
    assert np.count_nonzero(pred) == 4
    assert set(np.unique(pred)).issubset({0, 1, 2, 4})
