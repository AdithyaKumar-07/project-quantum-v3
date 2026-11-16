# 5. Project Structure and File Inventory

The companion `FILE_INVENTORY.csv` contains every file in this publication
build, with path, extension, size and a purpose category. Regenerate it after
adding/removing files.

## Top-level directories

| Directory | Responsibility |
|---|---|
| `app/` | Streamlit interface |
| `src/qmi/api/` | FastAPI application and API routes |
| `src/qmi/data/` | MSD/BraTS readers, preprocessing, splits and synthetic fixtures |
| `src/qmi/experiments/` | Experiment pipeline, resolution handling and runner |
| `src/qmi/quantum/` | FRQI/NEQR representations, simulation and patchwise workflows |
| `configs/` | Experiment defaults |
| `scripts/` | Dataset, training, quantum, calibration, export and validation entry points |
| `notebooks/` | Ordered research notebooks |
| `tests/` | Regression tests |
| `artifacts/` | Real MSD models, U-Net checkpoint, metadata and synthetic UI fixtures |
| `data/demo/` | Small synthetic demo data |
| `data/raw/` | Dataset placement directory; raw medical data is not included |
| `db/` | Runtime SQLite database location; local database is not included |
| `outputs/reports/` | CSV/JSON experimental evidence |
| `outputs/reports/quantum/` | FRQI/NEQR and high-resolution simulation outputs |
| `outputs/figures/` | Generated experiment and demo figures |
| `docs/` | Viewer-facing documentation |

## Important distinctions

- `msd_*` model artifacts are associated with real MSD training.
- `demo_*` and `notebook_demo_*` are synthetic fixtures for UI/testing.
- `*_uncalibrated*` files preserve pre-calibration evaluation; calibrated reports reflect validation-selected decision rules.
- `msd_unet_*` reports relate to the separate classical residual 2D U-Net.
- `msd_unet_vs_rf_*` reports are the paired held-out comparison; verify the paired CSVs before citing the comparison.
- Quantum `.npz` files contain numerical arrays; matching JSON files record metadata/metrics.
- `data/processed/unet_cache/`, the raw dataset, local database, backups and logs are deliberately excluded from this shareable build.

## Root-level files

The root README explains the research objective, architecture, major results,
limitations, quick start and links to the five viewer-facing documents.
`requirements*.txt` define Python dependencies. `pyproject.toml` defines
package/build configuration. `.env.example` is a template only; do not publish
real credentials or local `.env` files. `LICENSE` governs this project's
code and does not replace the dataset licence.
