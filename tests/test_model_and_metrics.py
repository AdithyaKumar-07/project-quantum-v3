import numpy as np
import pytest
from qmi.data.synthetic import make_synthetic_dataset
from qmi.models import balanced_pixel_samples, train_random_forest, predict_slice
from qmi.metrics import segmentation_metrics, reconstruction_metrics
from qmi.baselines import otsu_multiclass, KMeansMultiClassBaseline, WatershedMultiClassBaseline


def test_random_forest_train_predict_and_metrics():
    samples = make_synthetic_dataset(12, 48, 100)
    X, y = balanced_pixel_samples([(s.modalities, s.label) for s in samples[:8]], samples_per_class=100)
    model, report = train_random_forest(X, y, n_estimators=15, max_depth=10, n_jobs=1)
    pred = predict_slice(model, samples[8].modalities)
    assert pred.shape == samples[8].label.shape
    assert set(np.unique(pred)).issubset({0, 1, 2, 4})
    metrics = segmentation_metrics(pred, samples[8].label)
    assert 0 <= metrics["macro_dice"] <= 1
    assert report.n_samples == len(y)


def test_baselines_return_valid_label_maps():
    samples = make_synthetic_dataset(12, 48, 200)
    train = [(s.modalities, s.label) for s in samples[:8]]
    km = KMeansMultiClassBaseline().fit(train, max_pixels_per_slice=1000)
    ws = WatershedMultiClassBaseline(km)
    for sample in samples[8:10]:
        for pred in (otsu_multiclass(sample.modalities), km.predict(sample.modalities), ws.predict(sample.modalities)):
            assert pred.shape == sample.label.shape
            assert set(np.unique(pred)).issubset({0, 1, 2, 4})


def test_reconstruction_metrics_identity():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result = reconstruction_metrics(image, image, data_range=255)
    assert result["mse"] == 0
    assert np.isinf(result["psnr_db"])
    assert abs(result["ssim"] - 1.0) < 1e-10


def test_region_metrics_empty_mask_convention():
    empty = np.zeros((8, 8), dtype=np.uint8)
    result = segmentation_metrics(empty, empty)
    assert result["WT"]["dice"] == 1.0
    assert result["TC"]["iou"] == 1.0
    assert result["ET"]["specificity"] == 1.0
    assert np.isnan(result["WT"]["hd95_pixels"])
    assert np.isnan(result["WT"]["precision"])
    assert np.isnan(result["WT"]["recall"])
    assert result["WT"]["empty_ground_truth"] is True
    assert result["n_regions_both_empty"] == 3
    assert np.isnan(result["macro_dice_active"])


def test_hd95_uses_maximum_of_directed_surface_percentiles():
    truth = np.zeros((11, 11), dtype=np.uint8)
    pred = np.zeros_like(truth)
    truth[5, 5] = 2
    pred[5, 1:10] = 2
    result = segmentation_metrics(pred, truth)
    # The reverse directed distance is zero because the point lies on the line;
    # the prediction-to-truth 95th percentile is 4.0+ pixels. A pooled percentile
    # would be smaller and is not the symmetric maximum HD95 definition used here.
    assert result["WT"]["hd95_pixels"] == pytest.approx(4.0, abs=0.6)
