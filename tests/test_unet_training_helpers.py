import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
from scripts.train_unet_msd import validation_active_dice


class ClassOnePredictor(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros((x.shape[0], 4, x.shape[2], x.shape[3]), device=x.device)
        positive_signal = x[:, 0].mean(dim=(1, 2)) > 0.5
        logits[:, 0] = 5.0
        logits[positive_signal, 0] = -5.0
        logits[positive_signal, 1] = 5.0
        return logits


def test_validation_active_dice_is_patient_balanced_and_skips_both_empty_regions(tmp_path):
    patient = "case_a"
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    images = np.zeros((2, 4, 16, 16), dtype=np.float16)
    images[0, 0] = 1.0
    np.save(tmp_path / "images" / f"{patient}.npy", images)
    labels = np.zeros((2, 16, 16), dtype=np.uint8)
    labels[0] = 1
    np.save(tmp_path / "labels" / f"{patient}.npy", labels)
    score = validation_active_dice(ClassOnePredictor(), [patient], tmp_path, "cpu", 2)
    assert score == pytest.approx(1.0)
