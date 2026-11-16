import numpy as np
import pytest
from qmi.quantum.representations import (
    encode_frqi_state, encode_neqr_state, decode_frqi_state, decode_neqr_state,
    build_encoding_circuit,
)


@pytest.mark.parametrize("size", [16, 32])
def test_frqi_state_encoding_decoding_exact(size):
    image = np.random.default_rng(size).integers(0, 256, size=(size, size), dtype=np.uint8)
    state = encode_frqi_state(image)
    assert abs(np.linalg.norm(state) - 1.0) < 1e-10
    decoded = decode_frqi_state(state, size)
    np.testing.assert_array_equal(decoded, image)


@pytest.mark.parametrize("size", [16, 32])
def test_neqr_state_encoding_decoding_exact(size):
    image = np.random.default_rng(size + 1).integers(0, 256, size=(size, size), dtype=np.uint8)
    state = encode_neqr_state(image)
    assert abs(np.linalg.norm(state) - 1.0) < 1e-10
    decoded = decode_neqr_state(state, size)
    np.testing.assert_array_equal(decoded, image)


def test_quantum_state_qubit_requirements():
    assert int(np.log2(len(encode_frqi_state(np.zeros((16,16), dtype=np.uint8))))) == 9
    assert int(np.log2(len(encode_neqr_state(np.zeros((16,16), dtype=np.uint8))))) == 16
    assert int(np.log2(len(encode_frqi_state(np.zeros((32,32), dtype=np.uint8))))) == 11
    assert int(np.log2(len(encode_neqr_state(np.zeros((32,32), dtype=np.uint8))))) == 18


def test_invalid_image_resolution_is_rejected():
    with pytest.raises(ValueError):
        encode_frqi_state(np.zeros((8,8), dtype=np.uint8))


def test_qiskit_circuit_builder_requires_qiskit():
    try:
        import qiskit  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            build_encoding_circuit(np.zeros((16,16), dtype=np.uint8), "frqi")
    else:
        qc = build_encoding_circuit(np.zeros((16,16), dtype=np.uint8), "frqi")
        assert qc.num_qubits == 9
