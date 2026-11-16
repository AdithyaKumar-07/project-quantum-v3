"""Qiskit Aer ideal/noisy simulator adapters and circuit resource measurements."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class AerRun:
    reconstructed: np.ndarray
    counts: dict[str, int] | None
    statevector: np.ndarray | None
    metadata: dict


def _require_aer():
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
        from qiskit import transpile
    except ImportError as exc:
        raise ImportError("Qiskit Aer is required. Install with: pip install qiskit qiskit-aer") from exc
    return AerSimulator, NoiseModel, depolarizing_error, ReadoutError, transpile


def _build_noise_model(one_qubit_error: float, two_qubit_error: float, readout_error: float):
    AerSimulator, NoiseModel, depolarizing_error, ReadoutError, _ = _require_aer()
    for name, value in [("one_qubit_error", one_qubit_error), ("two_qubit_error", two_qubit_error), ("readout_error", readout_error)]:
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0,1)")
    noise = NoiseModel()
    basis = set(getattr(noise, "basis_gates", []))
    if one_qubit_error:
        err1 = depolarizing_error(one_qubit_error, 1)
        single_qubit_gates = [g for g in ("u", "u3", "sx", "x", "rz", "id") if g in basis]
        if not single_qubit_gates:
            single_qubit_gates = ["u"]
        noise.add_all_qubit_quantum_error(err1, single_qubit_gates)
    if two_qubit_error:
        err2 = depolarizing_error(two_qubit_error, 2)
        if "cx" not in basis:
            raise RuntimeError("Installed Aer NoiseModel does not expose the expected cx basis gate")
        noise.add_all_qubit_quantum_error(err2, ["cx"])
    if readout_error:
        ro = ReadoutError([[1-readout_error, readout_error], [readout_error, 1-readout_error]])
        noise.add_all_qubit_readout_error(ro)
    return noise


def _counts_to_image(counts: dict[str, int], representation: str, size: int) -> np.ndarray:
    n_position = 2 * int(np.log2(size))
    n_pixels = size * size
    sums = np.zeros(n_pixels, dtype=np.float64)
    totals = np.zeros(n_pixels, dtype=np.int64)
    for bitstring, count in counts.items():
        index = int(bitstring.replace(" ", ""), 2)
        pixel = index & (n_pixels - 1)
        if representation.lower() == "frqi":
            value = (index >> n_position) & 1
        else:
            value = index >> n_position
        sums[pixel] += float(value) * int(count)
        totals[pixel] += int(count)
    observed = totals > 0
    if not np.any(observed):
        raise RuntimeError("No measurement outcomes were returned by Aer")
    estimates = np.zeros(n_pixels, dtype=np.float64)
    estimates[observed] = sums[observed] / totals[observed]
    # If a finite-shot run misses a very rare position, fill from nearest observed pixel.
    if np.any(~observed):
        good = np.flatnonzero(observed)
        missing = np.flatnonzero(~observed)
        nearest = good[np.abs(missing[:, None] - good[None, :]).argmin(axis=1)]
        estimates[missing] = estimates[nearest]
    if representation.lower() == "frqi":
        # FRQI measurement estimates p(|1>) = sin(theta)^2. Recover the
        # encoded rotation angle before converting it back to grayscale.
        probability_one = np.clip(estimates, 0.0, 1.0)
        estimates = (2.0 / np.pi) * np.arcsin(np.sqrt(probability_one)) * 255.0
    else:
        estimates = estimates.clip(0, 255)
    return np.rint(estimates).astype(np.uint8).reshape(size, size)


def run_ideal(image: np.ndarray, representation: str, device: str = "CPU") -> AerRun:
    AerSimulator, _, _, _, transpile = _require_aer()
    from .representations import build_encoding_circuit, decode_frqi_state, decode_neqr_state
    qc = build_encoding_circuit(image, representation)
    run_circuit = qc.copy()
    run_circuit.save_statevector(label="statevector")
    requested_device = device.upper()
    fallback_reason = None

    def execute(selected_device: str):
        simulator = AerSimulator(method="statevector", device=selected_device)
        compiled_circuit = transpile(run_circuit, simulator, optimization_level=0)
        simulation_result = simulator.run(compiled_circuit).result()
        state = np.asarray(simulation_result.data(0)["statevector"], dtype=np.complex128)
        return compiled_circuit, state

    selected_device = requested_device
    try:
        compiled, state = execute(selected_device)
    except Exception as exc:
        if requested_device != "GPU":
            raise
        fallback_reason = f"GPU execution failed ({type(exc).__name__}: {exc}); CPU fallback used."
        selected_device = "CPU"
        compiled, state = execute("CPU")

    size = int(np.asarray(image).shape[0])
    recon = decode_frqi_state(state, size) if representation.lower() == "frqi" else decode_neqr_state(state, size)
    return AerRun(recon, None, state, {"mode": "ideal", "device": selected_device,
                                      "requested_device": requested_device,
                                      "n_qubits": qc.num_qubits, "circuit_depth_logical": qc.depth(),
                                      "compiled_depth": compiled.depth(), "compiled_ops": {str(k): int(v) for k, v in compiled.count_ops().items()},
                                      "fallback_reason": fallback_reason})


def run_noisy(image: np.ndarray, representation: str, shots: int | None = None,
              one_qubit_error: float = 0.001, two_qubit_error: float = 0.01,
              readout_error: float = 0.01) -> AerRun:
    AerSimulator, _, _, _, transpile = _require_aer()
    from .representations import build_encoding_circuit
    size = int(np.asarray(image).shape[0])
    n_pixels = size * size
    if shots is None:
        shots = max(20_000, n_pixels * 100)
    if shots < n_pixels:
        raise ValueError("shots should be at least the number of image pixels for a useful reconstruction")
    qc = build_encoding_circuit(image, representation)
    qc.measure_all()
    noise = _build_noise_model(one_qubit_error, two_qubit_error, readout_error)
    sim = AerSimulator(method="statevector", noise_model=noise, device="CPU")
    compiled = transpile(qc, sim, optimization_level=0)
    result = sim.run(compiled, shots=int(shots), seed_simulator=42).result()
    counts = result.get_counts(compiled)
    recon = _counts_to_image(counts, representation, size)
    return AerRun(recon, {str(k): int(v) for k, v in counts.items()}, None,
                  {"mode": "noisy_shot_based", "device": "CPU", "shots": int(shots),
                   "one_qubit_depolarizing_probability": float(one_qubit_error),
                   "two_qubit_depolarizing_probability": float(two_qubit_error),
                   "readout_error_probability": float(readout_error),
                   "n_qubits": qc.num_qubits, "compiled_depth": compiled.depth(),
                   "compiled_ops": {str(k): int(v) for k, v in compiled.count_ops().items()}})


def profile_circuit(image: np.ndarray, representation: str, max_qubits: int = 16) -> dict:
    AerSimulator, _, _, _, transpile = _require_aer()
    from .representations import build_encoding_circuit, encode_frqi_state, encode_neqr_state
    qc = build_encoding_circuit(image, representation)
    state = encode_frqi_state(image) if representation.lower() == "frqi" else encode_neqr_state(image)
    result = {"representation": representation.upper(), "n_qubits": int(qc.num_qubits),
              "statevector_dimension": int(len(state)), "nonzero_amplitudes": int(np.count_nonzero(state)),
              "logical_depth": int(qc.depth()), "logical_ops": {str(k): int(v) for k, v in qc.count_ops().items()},
              "transpilation_performed": False,
              "cost_model_note": "Initialize is a high-level exact state-preparation instruction; it is not free QRAM. Transpiled gate counts are reported only when transpilation completes."}
    if qc.num_qubits > max_qubits:
        result["transpilation_note"] = f"Skipped: {qc.num_qubits} qubits exceeds configured profiling cap {max_qubits}."
        return result
    sim = AerSimulator(method="statevector", device="CPU")
    compiled = transpile(qc, sim, optimization_level=0)
    result.update({"transpilation_performed": True, "transpiled_depth": int(compiled.depth()),
                   "transpiled_ops": {str(k): int(v) for k, v in compiled.count_ops().items()}})
    return result
