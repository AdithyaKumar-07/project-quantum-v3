import numpy as np
from qmi.features import extract_features


def test_global_coordinate_features_match_full_image_crop():
    rng = np.random.default_rng(9)
    full = rng.random((4, 20, 24), dtype=np.float32)
    full_features = extract_features(full).reshape(20, 24, -1)
    crop = full[:, 5:13, 7:18]
    crop_features = extract_features(crop, coordinate_origin=(5, 7),
                                     coordinate_full_shape=(20, 24)).reshape(8, 11, -1)
    np.testing.assert_allclose(crop_features[..., -2:], full_features[5:13, 7:18, -2:])
