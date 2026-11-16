from fastapi.testclient import TestClient
from qmi.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_request_accepts_native_resolution():
    from qmi.api.main import PredictionRequest
    request = PredictionRequest(case_id="BRATS_001", slice_index=10, resolution=240)
    assert request.resolution == 240


def test_prediction_request_rejects_unsupported_resolution():
    import pytest
    from pydantic import ValidationError
    from qmi.api.main import PredictionRequest
    with pytest.raises(ValidationError):
        PredictionRequest(case_id="BRATS_001", slice_index=10, resolution=64)


def test_prediction_request_defaults_to_random_forest():
    from qmi.api.main import PredictionRequest
    request = PredictionRequest(case_id="BRATS_001", slice_index=10, resolution=240)
    assert request.model_type == "rf"


def test_prediction_request_accepts_unet_only_at_native_resolution():
    from qmi.api.main import PredictionRequest
    request = PredictionRequest(case_id="BRATS_001", slice_index=10, resolution=240, model_type="unet")
    assert request.model_type == "unet"
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PredictionRequest(case_id="BRATS_001", slice_index=10, resolution=32, model_type="unet")
