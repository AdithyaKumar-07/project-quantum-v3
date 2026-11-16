import json
import numpy as np

from qmi.metrics import json_safe


def test_json_safe_converts_infinite_psnr_and_nan_metrics_to_null():
    payload = {
        "reconstruction": {"mse": 0.0, "psnr_db": float("inf"), "ssim": float("nan")},
        "numpy_inf": np.float64(float("inf")),
        "numpy_nan": np.float32(float("nan")),
    }
    safe = json_safe(payload)
    encoded = json.dumps(safe, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["reconstruction"]["mse"] == 0.0
    assert decoded["reconstruction"]["psnr_db"] is None
    assert decoded["reconstruction"]["ssim"] is None
    assert decoded["numpy_inf"] is None
    assert decoded["numpy_nan"] is None
