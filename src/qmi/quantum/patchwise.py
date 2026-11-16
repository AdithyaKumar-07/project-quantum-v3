"""Patch-wise high-resolution FRQI/NEQR reconstruction and native hybrid inference.

The complete high-resolution image is processed as non-overlapping 32x32 tiles.
Each tile uses the already-validated 32x32 FRQI/NEQR state representation. The
full-image statevector mode uses exact NumPy encode/decode; FRQI also supports a
full-grid Aer ideal mode. Optional Aer spot-checks are bounded and reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np

from ..models import predict_slice
from ..metrics import reconstruction_metrics, segmentation_metrics
from .representations import encode_frqi_state, encode_neqr_state, decode_frqi_state, decode_neqr_state

SUPPORTED_QUANTUM_RESOLUTIONS = (64, 128, 240)
SUPPORTED_TILE_SIZE = 32
SUPPORTED_MODES = ("statevector", "aer_ideal_full", "aer_ideal_spotcheck", "aer_noisy_spotcheck")


@dataclass
class PatchwiseQuantumResult:
    representation: str
    quantum_resolution: int
    tile_size: int
    reconstructed_flair: np.ndarray
    reconstructed_flair_native: np.ndarray
    predicted_labels_native: np.ndarray
    reconstruction: dict[str, float]
    segmentation: dict[str, Any] | None
    metadata: dict[str, Any]
    patch_records: list[dict[str, Any]]
    aer_tile_records: list[dict[str, Any]]
    total_seconds: float


def _resize_modalities(modalities: np.ndarray, size: int) -> np.ndarray:
    h, w = modalities.shape[1:]
    interpolation = cv2.INTER_AREA if size < min(h, w) else cv2.INTER_LINEAR
    return np.stack([cv2.resize(ch, (size, size), interpolation=interpolation)
                     for ch in modalities]).astype(np.float32)


def _pad_to_tiles(image: np.ndarray, tile_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[-2:]
    pad_h = (-h) % tile_size
    pad_w = (-w) % tile_size
    pads = ((0, 0), (0, pad_h), (0, pad_w))
    return np.pad(image, pads, mode="constant", constant_values=0), (pad_h, pad_w)


def _choose_spotcheck_tiles(truth: np.ndarray | None, grid: int, tile_size: int,
                            max_tiles: int) -> list[int]:
    if max_tiles < 1 or max_tiles > 4:
        raise ValueError("max_aer_tiles must be between 1 and 4")
    if truth is None:
        center = (grid // 2) * grid + (grid // 2)
        return [center]
    h, w = truth.shape
    candidates = []
    for row in range(grid):
        for col in range(grid):
            patch = truth[row*tile_size:min((row+1)*tile_size,h),
                          col*tile_size:min((col+1)*tile_size,w)]
            count = int(np.count_nonzero(patch))
            candidates.append((count, row*grid+col))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    positive = [idx for count, idx in candidates if count > 0]
    chosen = positive[:max_tiles]
    if len(chosen) < max_tiles:
        chosen.extend(idx for _, idx in candidates if idx not in chosen)
    return chosen[:max_tiles]


def run_patchwise_hybrid(native_modalities: np.ndarray, model, representation: str, *,
                         true_labels: np.ndarray | None = None,
                         quantum_resolution: int = 240,
                         tile_size: int = SUPPORTED_TILE_SIZE,
                         mode: str = "statevector",
                         max_aer_tiles: int = 1,
                         shots: int = 50_000,
                         one_qubit_error: float = 0.001,
                         two_qubit_error: float = 0.01,
                         readout_error: float = 0.01,
                         allow_expensive_18_qubit: bool = False) -> PatchwiseQuantumResult:
    """Reconstruct a 64/128/240 image in 32x32 quantum-representation tiles.

    ``statevector`` mode reconstructs every tile from exact NumPy state vectors.
    ``aer_ideal_full`` runs Qiskit Aer ideal FRQI for every tile and stitches the
    Aer outputs. Optional ideal/noisy spot-checks run on up to four tiles. The RF
    model is run once on the stitched, native-resolution four-modality image.
    """
    started = time.perf_counter()
    name = representation.strip().lower()
    if name not in {"frqi", "neqr"}:
        raise ValueError("representation must be FRQI or NEQR")
    if quantum_resolution not in SUPPORTED_QUANTUM_RESOLUTIONS:
        raise ValueError(f"quantum_resolution must be one of {SUPPORTED_QUANTUM_RESOLUTIONS}")
    if tile_size != SUPPORTED_TILE_SIZE:
        raise ValueError("This release uses the verified 32x32 quantum tile size")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {SUPPORTED_MODES}")
    if shots < tile_size * tile_size:
        raise ValueError("shots must be at least tile_size squared")
    if mode == "aer_ideal_full" and name != "frqi":
        raise ValueError("Full-grid Aer ideal execution is enabled for FRQI only. Use NEQR exact statevector full-grid plus a bounded Aer spot-check.")
    if any(not 0 <= x < 1 for x in (one_qubit_error, two_qubit_error, readout_error)):
        raise ValueError("Noise probabilities must lie in [0,1)")
    if mode != "statevector" and name == "neqr" and not allow_expensive_18_qubit:
        raise ValueError("32x32 NEQR Aer spot-check is an 18-qubit workload. Set allow_expensive_18_qubit=True explicitly; exact patchwise NEQR does not need this flag.")

    native = np.asarray(native_modalities, dtype=np.float32)
    if native.ndim != 3 or native.shape[0] != 4 or native.shape[1] != native.shape[2]:
        raise ValueError("native_modalities must be square with shape (4,H,W): FLAIR,T1,T1ce,T2")
    if not np.all(np.isfinite(native)) or np.any(native < 0) or np.any(native > 1):
        raise ValueError("native_modalities must be finite and normalized to [0,1]")
    native_h, native_w = native.shape[1:]
    if (native_h, native_w) != (240, 240):
        raise ValueError("The high-resolution hybrid path requires the native 240x240 MSD model input")
    if true_labels is not None:
        truth_native = np.asarray(true_labels, dtype=np.uint8)
        if truth_native.shape != (native_h, native_w):
            raise ValueError("true_labels must match native modality dimensions")
        if not np.all(np.isin(truth_native, [0, 1, 2, 4])):
            raise ValueError("true_labels must use internal labels {0,1,2,4}")
    else:
        truth_native = None

    target_modalities = native.copy() if quantum_resolution == native_h else _resize_modalities(native, quantum_resolution)
    reference = np.rint(target_modalities[0] * 255.0).clip(0,255).astype(np.uint8)
    padded_modalities, (pad_h, pad_w) = _pad_to_tiles(target_modalities, tile_size)
    padded_gray = np.rint(padded_modalities[0] * 255.0).clip(0,255).astype(np.uint8)
    padded_h, padded_w = padded_gray.shape
    grid_h, grid_w = padded_h // tile_size, padded_w // tile_size
    if grid_h != grid_w:
        raise ValueError("High-resolution quantum patch grid must be square")
    decoder = decode_frqi_state if name == "frqi" else decode_neqr_state
    encoder = encode_frqi_state if name == "frqi" else encode_neqr_state

    reconstructed_padded = np.zeros_like(padded_gray)
    patch_records: list[dict[str, Any]] = []
    full_aer_records: list[dict[str, Any]] = []
    full_aer_seconds = 0.0
    if mode == "aer_ideal_full":
        from .simulation import run_ideal
    for row in range(grid_h):
        for col in range(grid_w):
            y0, x0 = row * tile_size, col * tile_size
            tile = padded_gray[y0:y0+tile_size, x0:x0+tile_size]
            state = encoder(tile)
            tile_reconstruction = decoder(state, tile_size)
            if mode == "aer_ideal_full":
                aer_started = time.perf_counter()
                aer_tile = run_ideal(tile, name, device="CPU")
                aer_elapsed = time.perf_counter() - aer_started
                full_aer_seconds += aer_elapsed
                tile_reconstruction = np.asarray(aer_tile.reconstructed, dtype=np.uint8)
                full_aer_records.append({
                    "patch_index": row * grid_w + col, "row": row, "column": col,
                    "reconstruction": reconstruction_metrics(tile, tile_reconstruction, data_range=255),
                    "simulator_metadata": aer_tile.metadata, "wall_seconds": float(aer_elapsed),
                })
            reconstructed_padded[y0:y0+tile_size, x0:x0+tile_size] = tile_reconstruction
            patch_records.append({
                "patch_index": row * grid_w + col,
                "row": row, "column": col,
                "origin_y": y0, "origin_x": x0,
                "mse": reconstruction_metrics(tile, tile_reconstruction, data_range=255)["mse"],
                "statevector_dimension": int(state.size),
                "nonzero_amplitudes": int(np.count_nonzero(state)),
                "state_norm": float(np.linalg.norm(state)),
            })
            del state

    reconstructed = reconstructed_padded[:quantum_resolution, :quantum_resolution].copy()
    recon_metrics = reconstruction_metrics(reference, reconstructed, data_range=255)
    # Optional Aer spot-checks are isolated patch-level verification. They do not
    # replace the exact full-grid output and are never misreported as full-image Aer.
    aer_tile_records: list[dict[str, Any]] = []
    aer_seconds = 0.0
    if mode in {"aer_ideal_spotcheck", "aer_noisy_spotcheck"}:
        if truth_native is None:
            target_truth = None
        elif quantum_resolution == native_h:
            target_truth = truth_native
        else:
            target_truth = cv2.resize(truth_native, (quantum_resolution, quantum_resolution),
                                      interpolation=cv2.INTER_NEAREST)
        selected = _choose_spotcheck_tiles(target_truth, grid_h, tile_size, max_aer_tiles)
        from .simulation import run_ideal, run_noisy
        for patch_idx in selected:
            row, col = divmod(patch_idx, grid_w)
            y0, x0 = row * tile_size, col * tile_size
            patch = padded_gray[y0:y0+tile_size, x0:x0+tile_size]
            patch_started = time.perf_counter()
            if mode == "aer_ideal_spotcheck":
                aer_result = run_ideal(patch, name, device="CPU")
            else:
                aer_result = run_noisy(patch, name, shots=shots,
                    one_qubit_error=one_qubit_error,
                    two_qubit_error=two_qubit_error,
                    readout_error=readout_error)
            patch_seconds = time.perf_counter() - patch_started
            aer_seconds += patch_seconds
            aer_tile_records.append({
                "patch_index": int(patch_idx), "row": int(row), "column": int(col),
                "origin_y": int(y0), "origin_x": int(x0),
                "mode": mode, "reconstruction": reconstruction_metrics(patch, aer_result.reconstructed, data_range=255),
                "simulator_metadata": aer_result.metadata,
                "wall_seconds": float(patch_seconds),
            })

    # Resize the reconstructed quantum image back to native dimensions before
    # replacing FLAIR. T1/T1ce/T2 remain original and are never quantum-encoded.
    if quantum_resolution == native_h:
        reconstructed_native = reconstructed.copy()
    else:
        reconstructed_native = cv2.resize(reconstructed, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
    hybrid_native = native.copy()
    hybrid_native[0] = reconstructed_native.astype(np.float32) / 255.0
    predicted_native = predict_slice(model, hybrid_native)
    seg_metrics = segmentation_metrics(predicted_native, truth_native) if truth_native is not None else None

    metadata = {
        "workflow": "patchwise_high_resolution_quantum_reconstruction_then_native_rf_segmentation",
        "representation": name.upper(),
        "mode": mode,
        "backend_full_image": ("Qiskit Aer ideal statevector per 32x32 FRQI patch" if mode == "aer_ideal_full"
                               else "NumPy exact statevector encode/decode per 32x32 patch"),
        "backend_aer_spotcheck": "Qiskit Aer CPU" if mode in {"aer_ideal_spotcheck", "aer_noisy_spotcheck"} else None,
        "quantum_resolution": int(quantum_resolution),
        "native_resolution": [int(native_h), int(native_w)],
        "tile_size": int(tile_size),
        "grid": [int(grid_h), int(grid_w)],
        "total_patches": int(grid_h * grid_w),
        "padded_shape": [int(padded_h), int(padded_w)],
        "padding_bottom_right": [int(pad_h), int(pad_w)],
        "aer_spotcheck_tiles": int(len(aer_tile_records)),
        "aer_spotcheck_coverage_fraction": float(len(aer_tile_records) / (grid_h * grid_w)),
        "aer_spotcheck_seconds": float(aer_seconds),
        "aer_full_image_tile_runs": int(len(full_aer_records)),
        "aer_full_image_seconds": float(full_aer_seconds),
        "aer_results_are_not_used_for_full_image_reconstruction": mode != "aer_ideal_full",
        "native_other_modalities_preserved": True,
        "rf_inference_resolution": [int(native_h), int(native_w)],
        "note": ("FRQI full-image output is stitched from Qiskit Aer ideal simulation on every 32x32 tile. Qiskit Aer is a classical simulator; no quantum advantage is claimed."
                 if mode == "aer_ideal_full" else
                 "Exact state-vector patch reconstruction is a representation/decoder reference. Aer spot-check results are separate tile measurements, not whole-image Aer simulation. Qiskit Aer is a classical simulator; no quantum advantage is claimed."),
    }
    return PatchwiseQuantumResult(
        name.upper(), quantum_resolution, tile_size, reconstructed,
        reconstructed_native, predicted_native, recon_metrics, seg_metrics,
        metadata, patch_records, (full_aer_records if mode == "aer_ideal_full" else aer_tile_records),
        time.perf_counter() - started,
    )
