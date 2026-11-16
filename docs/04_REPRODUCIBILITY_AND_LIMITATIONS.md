# 4. Reproducibility and Limitations

## Reproducibility rules
1. Split subjects at patient level before extracting slices.
2. Preserve the saved `outputs/reports/msd_patient_split.json` for each run.
3. Select models, thresholds and calibration rules using training/validation data only.
4. Never tune thresholds against held-out test subjects.
5. Record exact CLI arguments, random seeds, software versions and model checkpoints.
6. Keep statevector, ideal Aer and finite-shot noisy Aer results separate.
7. Retain raw per-patient and per-slice CSV evidence.
8. Never report synthetic demo fixtures as experimental MSD outcomes.

## Local environment observed during validation
- Python 3.12.10
- NumPy 2.5.4; SciPy 1.18.1; pandas 3.1.0
- scikit-learn 1.8.0; scikit-image 0.26.0; nibabel 5.4.2
- OpenCV 5.0.0; Matplotlib 3.11.2; Plotly 7.1.0
- Streamlit 1.66.1; FastAPI 0.162.0; Uvicorn 0.54.0
- Qiskit 2.6.1; Qiskit Aer 0.18.2
- U-Net GPU run: NVIDIA RTX 4050 Laptop GPU, 6 GB; PyTorch reported CUDA 13.2.

The release environment check and test suite passed in the supplied local
console record (73 tests passed, with one Starlette/httpx deprecation warning).
Run the checks again on the exact revision being published.

## Limitations
- The segmentation task is 2D axial-slice processing; the U-Net is not a 3D volumetric model.
- The noisy FRQI/NEQR aggregate contains five held-out patients and is exploratory.
- Selected ideal/statevector high-resolution cases do not establish noisy full-image hardware execution.
- All quantum circuits were simulated on classical hardware.
- No physical QPU, quantum advantage or quantum speedup is demonstrated.
- HD95 is in 2D pixels, not millimetres.
- The U-Net is a research candidate, not clinically validated.
- TC and ET U-Net metrics show substantial patient-to-patient variability.
- The U-Net console metrics and paired comparison must be reconciled against the packaged CSVs before use in a formal paper.

## Data and privacy
The raw MSD dataset is not included in this publication package. Obtain it
separately and comply with the dataset terms and institutional requirements.
Local SQLite experiment history and processed per-patient caches are also
excluded from the public package by default.

## Citation and licensing
Retain the supplied project `LICENSE`. Dataset and third-party software
licences are separate from the project-code licence. Verify the official
dataset citation, attribution and redistribution terms before public release.
