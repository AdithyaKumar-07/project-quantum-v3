import numpy as np
import pytest

from qmi.quantum.patchwise import run_patchwise_hybrid


class _BackgroundModel:
    def predict(self, features):
        return np.zeros(features.shape[0], dtype=np.uint8)


def _native_fixture(size=240):
    rng = np.random.default_rng(123)
    modalities = rng.random((4, size, size), dtype=np.float32)
    truth = np.zeros((size, size), dtype=np.uint8)
    truth[size//2-8:size//2+8, size//2-8:size//2+8] = 2
    return modalities, truth


def test_highres_frqi_64_patchwise_exact_reconstruction_and_native_segmentation():
    modalities, truth = _native_fixture()
    result = run_patchwise_hybrid(modalities, _BackgroundModel(), "FRQI",
                                  true_labels=truth, quantum_resolution=64)
    assert result.reconstructed_flair.shape == (64, 64)
    assert result.reconstructed_flair_native.shape == (240, 240)
    assert result.predicted_labels_native.shape == (240, 240)
    assert result.metadata["total_patches"] == 4
    assert result.metadata["backend_full_image"].startswith("NumPy exact statevector")
    assert result.metadata["aer_results_are_not_used_for_full_image_reconstruction"] is True
    assert result.reconstruction["mse"] == 0.0
    assert result.segmentation is not None


def test_highres_neqr_64_patchwise_exact_reconstruction():
    modalities, truth = _native_fixture()
    result = run_patchwise_hybrid(modalities, _BackgroundModel(), "NEQR",
                                  true_labels=truth, quantum_resolution=64)
    assert result.reconstructed_flair.shape == (64, 64)
    assert result.metadata["total_patches"] == 4
    assert result.reconstruction["mse"] == 0.0


def test_native_240_padding_and_patch_count():
    modalities, truth = _native_fixture()
    result = run_patchwise_hybrid(modalities, _BackgroundModel(), "FRQI",
                                  true_labels=truth, quantum_resolution=240)
    assert result.reconstructed_flair.shape == (240, 240)
    assert result.reconstructed_flair_native.shape == (240, 240)
    assert result.metadata["grid"] == [8, 8]
    assert result.metadata["total_patches"] == 64
    assert result.metadata["padding_bottom_right"] == [16, 16]
    assert result.reconstruction["mse"] == 0.0


def test_highres_neqr_aer_requires_explicit_resource_opt_in():
    modalities, truth = _native_fixture()
    with pytest.raises(ValueError, match="18-qubit"):
        run_patchwise_hybrid(modalities, _BackgroundModel(), "NEQR",
                             true_labels=truth, quantum_resolution=64,
                             mode="aer_ideal_spotcheck")


def test_highres_invalid_resolution_is_rejected():
    modalities, truth = _native_fixture()
    with pytest.raises(ValueError, match="quantum_resolution"):
        run_patchwise_hybrid(modalities, _BackgroundModel(), "FRQI",
                             true_labels=truth, quantum_resolution=96)


def test_highres_128_patchwise_frqi_has_sixteen_tiles():
    modalities, truth = _native_fixture()
    result = run_patchwise_hybrid(modalities, _BackgroundModel(), "FRQI",
                                  true_labels=truth, quantum_resolution=128)
    assert result.metadata["total_patches"] == 16
    assert result.reconstructed_flair.shape == (128, 128)
    assert result.reconstruction["mse"] == 0.0


def test_aer_spotcheck_is_separate_from_full_exact_image(monkeypatch):
    import qmi.quantum.simulation as simulation
    from qmi.quantum.simulation import AerRun

    def fake_ideal(image, representation, device="CPU"):
        return AerRun(np.asarray(image, dtype=np.uint8).copy(), None, None,
                      {"mode": "ideal", "device": "CPU", "stub": True})

    monkeypatch.setattr(simulation, "run_ideal", fake_ideal)
    modalities, truth = _native_fixture()
    result = run_patchwise_hybrid(modalities, _BackgroundModel(), "FRQI",
        true_labels=truth, quantum_resolution=64,
        mode="aer_ideal_spotcheck", max_aer_tiles=1)
    assert result.reconstruction["mse"] == 0.0
    assert len(result.aer_tile_records) == 1
    assert result.aer_tile_records[0]["simulator_metadata"]["stub"] is True
    assert result.metadata["aer_spotcheck_coverage_fraction"] == 0.25
    assert result.metadata["aer_results_are_not_used_for_full_image_reconstruction"] is True


def test_aer_ideal_full_runs_every_frqi_tile(monkeypatch):
    import qmi.quantum.simulation as simulation
    from qmi.quantum.simulation import AerRun
    calls = []

    def fake_ideal(image, representation, device="CPU"):
        calls.append((np.asarray(image).shape, representation, device))
        return AerRun(np.asarray(image, dtype=np.uint8).copy(), None, None,
                      {"mode": "ideal", "device": "CPU", "tile_call": len(calls)})

    monkeypatch.setattr(simulation, "run_ideal", fake_ideal)
    modalities, truth = _native_fixture()
    result = run_patchwise_hybrid(modalities, _BackgroundModel(), "FRQI",
        true_labels=truth, quantum_resolution=64, mode="aer_ideal_full")
    assert len(calls) == 4
    assert len(result.aer_tile_records) == 4  # one record per full-grid Aer tile
    assert result.metadata["aer_full_image_tile_runs"] == 4
    assert result.metadata["aer_results_are_not_used_for_full_image_reconstruction"] is False
    assert result.reconstruction["mse"] == 0.0


def test_full_grid_aer_ideal_is_frqi_only():
    modalities, truth = _native_fixture()
    with pytest.raises(ValueError, match="FRQI only"):
        run_patchwise_hybrid(modalities, _BackgroundModel(), "NEQR",
                             true_labels=truth, quantum_resolution=64,
                             mode="aer_ideal_full")


def test_native_240_neqr_full_patchwise_statevector():
    modalities, truth = _native_fixture()
    result = run_patchwise_hybrid(modalities, _BackgroundModel(), "NEQR",
                                  true_labels=truth, quantum_resolution=240)
    assert result.metadata["total_patches"] == 64
    assert result.metadata["grid"] == [8, 8]
    assert result.reconstruction["mse"] == 0.0
    assert result.segmentation is not None
