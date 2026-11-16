# 1. Installation and Usage

## Requirements
- Windows 10/11, 64-bit
- Python 3.12, 64-bit
- The separately obtained MSD Task01_BrainTumour dataset for full data workflows
- NVIDIA GPU is optional for the classical Random Forest/quantum simulator workflows; PyTorch U-Net GPU training requires a compatible CUDA-enabled PyTorch installation.

The validated development environment reported Python 3.12.10, NumPy 2.5.4,
SciPy 1.18.1, pandas 3.1.0, scikit-learn 1.8.0, scikit-image 0.26.0,
OpenCV 5.0.0, nibabel 5.4.2, Matplotlib 3.11.2, Plotly 7.1.0,
Streamlit 1.66.1, FastAPI 0.162.0, Uvicorn 0.54.0, Qiskit 2.6.1
and Qiskit Aer 0.18.2. Versions in a fresh installation are controlled by
the requirements files.

## Dataset placement
Obtain MSD Task01_BrainTumour separately. Do not commit or redistribute the
raw MRI dataset in this repository unless the dataset licence and institutional
data rules explicitly permit it.

Extract to:

```text
data/raw/Task01_BrainTumour/
    dataset.json
    imagesTr/
    labelsTr/
    imagesTs/
```

Keep the original NIfTI files unchanged. The expected labelled training
cohort in the validated local dataset was 484 image/label pairs.

## Install
From PowerShell in the repository root:

```powershell
.\setup_windows.bat
```

If you prefer manual setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python scripts/check_environment.py
```

Optional U-Net dependencies are listed in `requirements-unet.txt`.
Quantum packages are pinned in `requirements-quantum.txt` and are also
included in the main environment as configured by the project.

## Verify
```powershell
python scripts/verify_dataset.py
python -m pytest -q
python scripts/validate_release.py --require-dataset --require-models --require-reports
```

## Launch the application
```powershell
.\run_all.bat
```

The launcher checks the API on port 8000 and starts/reuses the correct
Project Quantum API before launching Streamlit. Do not start a duplicate
API if port 8000 is already serving this project.

Addresses:
- Streamlit: `http://localhost:8501`
- API health: `http://127.0.0.1:8000/api/health`
- API documentation: `http://127.0.0.1:8000/docs`
- Dataset status: `http://127.0.0.1:8000/api/dataset/status`

To run separately, use two terminals:
```powershell
.\run_api.bat
```
```powershell
.\run_frontend.bat
```

## Notebooks
Open JupyterLab using:
```powershell
.\run_notebook.bat
```

Recommended order:
1. `01_dataset_exploration_and_quality_control.ipynb`
2. `02_patient_split_training_validation_testing.ipynb`
3. `03_classical_baseline_metrics_and_visualization.ipynb`
4. `04_frqi_neqr_quantum_hybrid_lab.ipynb`
5. `05_results_export_and_research_tables.ipynb`
6. `06_unet_model_training_and_evaluation.ipynb`

The consolidated CLI scripts are the canonical experiment path; notebooks
are research and analysis companions. Do not regenerate the patient split
unless intentionally starting a new experiment.

## Retraining
Classical MSD models:
```powershell
python scripts/train_msd.py --max-patients 0 --resolution all --slices-per-patient 8 --samples-per-class 250 --trees 120 --seed 42
```

FRQI/NEQR ideal study:
```powershell
python scripts/run_quantum_study.py --resolutions both --mode ideal --max-cases 5
```

FRQI/NEQR noisy study:
```powershell
python scripts/run_quantum_study.py --resolutions 16 --mode noisy --shots 50000 --max-cases 5
```

Validation-only calibration and report regeneration:
```powershell
python scripts/finalize_best_model.py --resolutions all --slices-per-patient 8
```

Residual 2D U-Net training:
```powershell
python scripts/train_unet_msd.py --device cuda
```

The last command is a training entry point, not a requirement for running
the dashboard when the provided checkpoint is available. Review its `--help`
for all supported arguments before starting another training run.
