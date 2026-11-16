"""Exact FRQI and NEQR state-vector representations and Qiskit circuit builders.

Qubit ordering: position register occupies the least-significant 2*log2(N) qubits;
intensity register occupies the remaining most-significant qubits.
"""
from __future__ import annotations
import numpy as np


def _validate_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Quantum image input must be a square 2D grayscale image")
    size = arr.shape[0]
    if size not in (16, 32):
        raise ValueError("Only the locked project resolutions 16x16 and 32x32 are supported")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Image contains NaN or infinite values")
    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() < 0 or arr.max() > 1.0 + 1e-6:
            raise ValueError("Floating-point images must be normalized to [0,1]")
        gray = np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    else:
        if arr.min() < 0 or arr.max() > 255:
            raise ValueError("Integer grayscale values must be in [0,255]")
        gray = arr.astype(np.uint8)
    return gray


def encode_frqi_state(image: np.ndarray) -> np.ndarray:
    """Create the exact FRQI state vector for a 16x16 or 32x32 grayscale image."""
    gray = _validate_image(image)
    size = gray.shape[0]
    n_position = 2 * int(np.log2(size))
    n_qubits = n_position + 1
    n_pixels = size * size
    theta = (np.pi / 2.0) * (gray.astype(np.float64).ravel() / 255.0)
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    scale = 1.0 / np.sqrt(n_pixels)
    positions = np.arange(n_pixels, dtype=np.int64)
    state[positions] = scale * np.cos(theta)                 # intensity qubit |0>
    state[(1 << n_position) + positions] = scale * np.sin(theta)  # intensity qubit |1>
    return state


def encode_neqr_state(image: np.ndarray) -> np.ndarray:
    """Create the exact NEQR state vector with 8-bit computational-basis intensity."""
    gray = _validate_image(image)
    size = gray.shape[0]
    n_position = 2 * int(np.log2(size))
    n_qubits = n_position + 8
    n_pixels = size * size
    positions = np.arange(n_pixels, dtype=np.int64)
    basis_indices = positions + (gray.astype(np.int64).ravel() << n_position)
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    state[basis_indices] = 1.0 / np.sqrt(n_pixels)
    return state


def decode_frqi_state(statevector: np.ndarray, size: int) -> np.ndarray:
    n_position = 2 * int(np.log2(size))
    expected = 1 << (n_position + 1)
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    if len(state) != expected:
        raise ValueError(f"Expected a {expected}-amplitude FRQI statevector, got {len(state)}")
    positions = np.arange(size * size)
    a0 = np.abs(state[positions])
    a1 = np.abs(state[(1 << n_position) + positions])
    theta = np.arctan2(a1, a0)
    gray = np.rint((2.0 / np.pi) * theta * 255.0).clip(0, 255).astype(np.uint8)
    return gray.reshape(size, size)


def decode_neqr_state(statevector: np.ndarray, size: int) -> np.ndarray:
    n_position = 2 * int(np.log2(size))
    expected = 1 << (n_position + 8)
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    if len(state) != expected:
        raise ValueError(f"Expected a {expected}-amplitude NEQR statevector, got {len(state)}")
    probabilities = np.abs(state) ** 2
    positions = np.arange(size * size, dtype=np.int64)
    # Exact basis-state decode; expectation is also well-defined for noisy superpositions.
    values = np.zeros(size * size, dtype=np.float64)
    position_probability = np.zeros(size * size, dtype=np.float64)
    for intensity in range(256):
        indices = positions + (intensity << n_position)
        p = probabilities[indices]
        values += intensity * p
        position_probability += p
    values = np.divide(values, position_probability, out=np.zeros_like(values), where=position_probability > 0)
    return np.rint(values.clip(0, 255)).astype(np.uint8).reshape(size, size)


def build_encoding_circuit(image: np.ndarray, representation: str):
    """Build a Qiskit state-preparation circuit for FRQI or NEQR.

    This uses exact amplitude/state preparation. It is a valid encoding baseline but
    does not assume free QRAM or an efficient hardware data-loading oracle. The cost
    of transpiled state preparation is measured rather than hidden.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError("Qiskit is required. Install with: pip install qiskit qiskit-aer") from exc
    name = representation.strip().lower()
    if name == "frqi":
        state = encode_frqi_state(image)
    elif name == "neqr":
        state = encode_neqr_state(image)
    else:
        raise ValueError("representation must be 'frqi' or 'neqr'")
    n_qubits = int(np.log2(len(state)))
    qc = QuantumCircuit(n_qubits, name=f"{name.upper()}_state_preparation")
    qc.initialize(state, list(range(n_qubits)))
    qc.metadata = {"representation": name.upper(), "image_size": int(np.asarray(image).shape[0]),
                   "encoding": "exact_state_preparation", "n_qubits": n_qubits}
    return qc
