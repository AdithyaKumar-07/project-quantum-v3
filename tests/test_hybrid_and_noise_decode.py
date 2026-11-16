import numpy as np

from qmi.hybrid import run_hybrid_path
from qmi.quantum.simulation import AerRun, _counts_to_image


class _BackgroundModel:
    def predict(self, features):
        return np.zeros(features.shape[0], dtype=np.uint8)


def test_hybrid_pipeline_uses_reconstructed_flair(monkeypatch):
    import qmi.hybrid as hybrid

    rng = np.random.default_rng(7)
    modalities = rng.random((4, 16, 16), dtype=np.float32)
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[4:8, 4:8] = 2
    expected = np.rint(modalities[0] * 255).astype(np.uint8)

    def fake_ideal(image, representation, device="CPU"):
        return AerRun(
            reconstructed=np.asarray(image, dtype=np.uint8).copy(),
            counts=None,
            statevector=None,
            metadata={"mode": "ideal", "device": "CPU", "test_stub": True},
        )

    monkeypatch.setattr(hybrid, "run_ideal", fake_ideal)
    result = run_hybrid_path(modalities, _BackgroundModel(), "FRQI", true_labels=labels)

    assert result.representation == "FRQI"
    assert result.reconstructed_flair.shape == (16, 16)
    np.testing.assert_array_equal(result.reconstructed_flair, expected)
    assert result.predicted_labels.shape == labels.shape
    assert result.segmentation is not None
    assert result.simulator_metadata["test_stub"] is True
    assert result.simulation_seconds >= 0
    assert result.inference_seconds >= 0
    assert result.total_seconds >= result.inference_seconds


def test_frqi_finite_shot_decode_inverts_sine_squared_probability():
    # For p(|1>)=0.25, theta=asin(sqrt(.25))=pi/6, which maps to 85/255.
    size = 16
    n_position = 2 * int(np.log2(size))
    counts = {}
    for position in range(size * size):
        bit0 = format(position, f"0{n_position + 1}b")
        bit1 = format(position + (1 << n_position), f"0{n_position + 1}b")
        counts[bit0] = 75
        counts[bit1] = 25
    decoded = _counts_to_image(counts, "frqi", size)
    np.testing.assert_array_equal(decoded, np.full((size, size), 85, dtype=np.uint8))
