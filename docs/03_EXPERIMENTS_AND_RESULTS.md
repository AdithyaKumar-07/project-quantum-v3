# 3. Experiments and Results

## Classical Random Forest resolution study
Reported held-out values:

| Resolution | Conventional macro Dice | Active-region macro Dice | Active-region macro IoU |
|---|---:|---:|---:|
| 16 × 16 | ≈0.594 | 0.4155 ± 0.3255 | ≈0.337 |
| 32 × 32 | ≈0.591 | 0.4538 ± 0.3292 | 0.3678 |
| 240 × 240 | ≈0.477 | 0.4365 ± 0.3219 | 0.3473 |

16 × 16 has the highest conventional macro Dice; 32 × 32 has the
highest active-region macro Dice and IoU. Report the metric convention
whenever identifying a best resolution. Results are slice-based, not
volumetric Dice.

## Five-patient noisy FRQI/NEQR comparison
Configuration: 16 × 16, five held-out patients, 50,000 shots.

| Metric | FRQI | NEQR |
|---|---:|---:|
| Mean reconstruction MSE | 330.4227 | 30.6727 |
| Active-region Dice | 0.5490 | 0.5305 |
| Recorded runtime | 2.84 s | 172.72 s |

NEQR produced lower reconstruction error in this selected experiment.
FRQI produced slightly higher downstream Dice and a shorter recorded
simulator runtime. NEQR's runtime was about 60.8 times the FRQI runtime
under this specific configuration. This is not a general complexity result
or evidence of a quantum speedup.

## Selected ideal/high-resolution reconstruction
The project records exact statevector patchwise reconstruction for selected
64 × 64, 128 × 128 and 240 × 240 cases. A complete ideal 240 × 240 tiled
FRQI Aer run completed 64 tiles with MSE 0, SSIM 1.0 and recorded runtime
4.86 s. The 240 × 240 noisy FRQI Aer experiment was a one-tile spot check,
not a full-image noisy run. Full native-resolution NEQR Aer execution was
not recorded as completed.

## Recorded GPU residual 2D U-Net
The recorded compact GPU run used:
- NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)
- 10 epochs; batch size 2
- 2 slices per patient; base channels 8
- Learning rate 0.001; early-stopping patience 3
- Random seed 42; 338/72/74 patient split

Recorded held-out results:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Macro Dice | 0.743423 | 0.230548 |
| Macro IoU | 0.660040 | 0.226139 |
| Active-region macro Dice | 0.732803 | 0.234538 |
| Active-region macro IoU | 0.645767 | 0.229133 |
| WT Dice | 0.869696 | 0.144771 |
| TC Dice | 0.667548 | 0.333244 |
| ET Dice | 0.693023 | 0.330239 |

The paired held-out summary recorded the following comparison:

| Metric | Random Forest | Residual 2D U-Net |
|---|---:|---:|
| Macro Dice | 0.427752 | 0.743423 |
| Macro IoU | 0.314343 | 0.660040 |
| Active-region macro Dice | 0.427752 | 0.732803 |
| Active-region macro IoU | 0.314343 | 0.645767 |
| WT Dice | 0.462928 | 0.869696 |
| TC Dice | 0.406219 | 0.667548 |
| ET Dice | 0.414109 | 0.693023 |

The training record and console output are the basis for these values.
Before publication, reconcile all entries with:
- `outputs/reports/msd_unet_test_summary.csv`
- `outputs/reports/msd_unet_test_per_patient.csv`
- `outputs/reports/msd_unet_training_history.csv`
- `outputs/reports/msd_unet_vs_rf_paired_test_summary.csv`
- the corresponding per-patient and per-slice paired CSVs.

Do not combine the paired RF comparison row with the separate multi-resolution
RF table without explaining the matching protocol and metric aggregation.

## Report locations
- `outputs/reports/`: classical and U-Net CSV/JSON summaries
- `outputs/reports/quantum/`: FRQI/NEQR experiments and resource profiles
- `outputs/reports/quantum/highres/`: selected high-resolution patchwise records
- `outputs/figures/`: generated visuals

Files prefixed `demo_` or `notebook_demo_` are synthetic fixtures for UI
demonstration/testing. They are not real MSD results.
