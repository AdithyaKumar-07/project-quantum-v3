import numpy as np
from qmi.metrics import segmentation_metrics, reconstruction_metrics


def test_nested_region_dice_for_perfect_prediction():
    truth = np.array([[0, 1, 2, 4], [0, 0, 2, 4]], dtype=np.uint8)
    result = segmentation_metrics(truth.copy(), truth)
    assert result["WT"]["dice"] == 1.0
    assert result["TC"]["dice"] == 1.0
    assert result["ET"]["dice"] == 1.0
    assert result["macro_dice"] == 1.0


def test_reconstruction_metrics_exact_image():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result = reconstruction_metrics(image, image.copy(), data_range=255)
    assert result["mse"] == 0.0
    assert result["psnr_db"] == float("inf")
    assert abs(result["ssim"] - 1.0) < 1e-12
