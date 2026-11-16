"""Hybrid quantum-classical path: encode/reconstruct FLAIR, then segment with the trained RF."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .models import predict_slice
from .metrics import reconstruction_metrics, segmentation_metrics
from .quantum.simulation import run_ideal, run_noisy

@dataclass
class HybridResult:
    representation: str
    simulation_mode: str
    reconstructed_flair: np.ndarray
    predicted_labels: np.ndarray
    reconstruction: dict
    segmentation: dict | None
    simulator_metadata: dict
    simulation_seconds: float
    inference_seconds: float
    total_seconds: float


def run_hybrid_path(modalities: np.ndarray, model, representation: str, *,
                    true_labels: np.ndarray | None = None, simulation_mode: str = "ideal",
                    shots: int | None = None, device: str = "CPU",
                    noise_parameters: dict | None = None) -> HybridResult:
    """Run FRQI/NEQR on FLAIR and feed reconstructed FLAIR + original other modalities to RF.

    `modalities` order is FLAIR, T1, T1ce, T2 and must be (4,16,16) or (4,32,32).
    The classical modalities are held fixed so the experiment isolates the effect of
    the quantum representation/reconstruction of the primary FLAIR image.
    """
    start = time.perf_counter()
    m = np.asarray(modalities, dtype=np.float32)
    if m.ndim != 3 or m.shape[0] != 4 or m.shape[1] != m.shape[2] or m.shape[1] not in (16, 32):
        raise ValueError("Expected four aligned modalities with shape (4,16,16) or (4,32,32)")
    if not np.all(np.isfinite(m)) or np.any(m < 0) or np.any(m > 1):
        raise ValueError("Modalities must contain finite normalized values in [0,1]")
    if true_labels is not None and np.asarray(true_labels).shape != m.shape[1:]:
        raise ValueError("Ground-truth label map must match modality dimensions")
    gray8 = np.rint(m[0] * 255.0).clip(0,255).astype(np.uint8)
    mode = simulation_mode.strip().lower()
    params = noise_parameters or {}
    simulation_start = time.perf_counter()
    if mode == "ideal":
        aer_result = run_ideal(gray8, representation, device=device)
    elif mode == "noisy":
        aer_result = run_noisy(gray8, representation, shots=shots,
            one_qubit_error=float(params.get("one_qubit_error", 0.001)),
            two_qubit_error=float(params.get("two_qubit_error", 0.01)),
            readout_error=float(params.get("readout_error", 0.01)))
    else:
        raise ValueError("simulation_mode must be 'ideal' or 'noisy'")
    simulation_seconds = time.perf_counter() - simulation_start
    reconstructed = np.asarray(aer_result.reconstructed, dtype=np.uint8)
    hybrid_input = m.copy()
    hybrid_input[0] = reconstructed.astype(np.float32) / 255.0
    inference_start = time.perf_counter()
    predicted = predict_slice(model, hybrid_input)
    inference_seconds = time.perf_counter() - inference_start
    recon_scores = reconstruction_metrics(gray8, reconstructed, data_range=255)
    seg_scores = segmentation_metrics(predicted, true_labels) if true_labels is not None else None
    return HybridResult(representation.upper(), mode, reconstructed, predicted, recon_scores,
                        seg_scores, aer_result.metadata, simulation_seconds, inference_seconds,
                        time.perf_counter() - start)
