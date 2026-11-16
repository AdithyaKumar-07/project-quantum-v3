# 2. System Architecture and Methodology

## System overview
The application has a Streamlit presentation layer and a FastAPI backend.
Reusable scientific functionality is maintained in the `qmi` Python package.

```text
Streamlit UI (app/)
       |
       v
FastAPI (src/qmi/api/)
       |
       +--> Data adapters and preprocessing (src/qmi/data/)
       +--> Classical models and features
       +--> FRQI/NEQR encoders and simulator
       +--> Experiment runner and metrics
       +--> SQLite experiment history
       |
       v
Model artifacts + CSV/JSON reports + research figures
```

## Dataset
The experiments use Medical Segmentation Decathlon Task01_BrainTumour,
a BraTS 2016/2017-derived release, not BraTS 2021. The labelled cohort
contains 484 subjects with four co-registered MRI modalities:
FLAIR, T1, T1ce and T2. In-plane dimensions are 240 × 240 with
155 axial slices.

Source label IDs:
- 0: background
- 1: necrotic/non-enhancing tumour core
- 2: oedema
- 4: enhancing tumour

Evaluation regions:
- Whole Tumour (WT) = {1,2,4}
- Tumour Core (TC) = {1,4}
- Enhancing Tumour (ET) = {4}

The documented patient split is 338 training, 72 validation and
74 held-out test subjects (seed 42). Splitting occurs at patient level
before slice extraction to avoid leakage from correlated adjacent slices.

## Preprocessing
Images use finite, non-zero voxel intensity normalization with percentile
clipping. At 16 × 16 and 32 × 32, image resizing uses area interpolation.
Label resizing uses a class-occupancy rule: tumour-positive cells are
assigned the most represented tumour class; otherwise they are background.
This helps preserve small positive classes but can enlarge their geometry.
Native 240 × 240 labels are retained without this downsampling.

## FRQI
For an N × N grayscale image, FRQI uses 2 log2(N) position qubits and
one intensity qubit. For 8-bit grayscale intensity g_p in [0,255]:

theta_p = (pi/2) * (g_p/255)

|I_FRQI> = (1/N) sum_p |p> (cos(theta_p)|0> + sin(theta_p)|1>)

Register sizes:
- 16 × 16: 9 qubits
- 32 × 32: 11 qubits

## NEQR
NEQR stores the 8-bit grayscale value in computational-basis qubits:

|I_NEQR> = (1/N) sum_p |g_p>|p>

Total qubits = 2 log2(N) + 8:
- 16 × 16: 16 qubits
- 32 × 32: 18 qubits

## Hybrid FRQI/NEQR path
Only FLAIR enters the quantum representation/reconstruction path. The
reconstructed FLAIR is combined with unchanged T1, T1ce and T2 channels.
A classical pixel-wise Random Forest performs the downstream segmentation.
The quantum circuits do not directly predict tumour labels.

## Separate residual 2D U-Net
The residual U-Net is a separate classical multi-modal model at native
240 × 240 resolution. It is not part of the FRQI/NEQR circuit and is not
a 3D volumetric network.

## Simulation modes
The project separates:
1. Mathematical/NumPy statevector encode–decode round trips.
2. Ideal Qiskit Aer circuit simulations.
3. Finite-shot noisy Aer simulations.

Qiskit Aer is a classical simulator. Statevector access is not equivalent
to reading every amplitude from a physical quantum processor. Circuit
preparation, data loading, sampling, measurement and reconstruction costs
must be considered in any resource comparison.
