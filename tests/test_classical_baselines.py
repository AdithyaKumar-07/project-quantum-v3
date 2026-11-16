import numpy as np
from qmi.baselines import KMeansMultiClassBaseline, WatershedMultiClassBaseline, otsu_multiclass
from qmi.data.synthetic import make_synthetic_dataset


def test_classical_baselines_return_valid_internal_labels():
    samples = make_synthetic_dataset(n_samples=8, size=32, seed=101)
    training = [(s.modalities, s.label) for s in samples[:6]]
    test_image = samples[6].modalities
    kmeans = KMeansMultiClassBaseline(random_state=7).fit(training, max_pixels_per_slice=500)
    predictions = [otsu_multiclass(test_image), kmeans.predict(test_image), WatershedMultiClassBaseline(kmeans).predict(test_image)]
    for prediction in predictions:
        assert prediction.shape == test_image.shape[1:]
        assert set(np.unique(prediction)).issubset({0, 1, 2, 4})
