"""Optional small integration smoke test; skipped when Qiskit Aer is unavailable."""
import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qmi.quantum.simulation import run_ideal


def test_frqi_16_aer_ideal_roundtrip():
    image = np.random.default_rng(2026).integers(0, 256, size=(16, 16), dtype=np.uint8)
    result = run_ideal(image, "FRQI", device="CPU")
    np.testing.assert_array_equal(result.reconstructed, image)
    assert result.metadata["mode"] == "ideal"
    assert result.metadata["device"] == "CPU"
    assert result.metadata["n_qubits"] == 9
