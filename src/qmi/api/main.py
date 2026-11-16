"""FastAPI service for dataset status, model registry, inference and run history."""
from __future__ import annotations
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qmi.db import init_db, list_experiments, list_predictions, add_experiment, add_prediction_event
from qmi.data.msd_dataset import read_msd_metadata, discover_msd_cases, load_msd_case
from qmi.data.preprocessing import resize_slice
from qmi.models import load_model, predict_slice
from qmi.metrics import segmentation_metrics, json_safe

@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Project Quantum V3.4 API",
    version="3.4.0",
    description="Research API for hybrid quantum-classical MRI image processing and segmentation.",
    lifespan=lifespan,
)


class ExperimentRecord(BaseModel):
    experiment_name: str = Field(min_length=2, max_length=160)
    dataset_name: str = Field(min_length=2, max_length=240)
    status: str = Field(default="completed", max_length=40)
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_dir: str | None = None


class PredictionRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=160)
    slice_index: int = Field(ge=0)
    resolution: int = Field(ge=16, le=240)
    model_type: str = Field(default="rf")

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: int) -> int:
        if value not in (16, 32, 240):
            raise ValueError("resolution must be 16, 32 or native 240")
        return value

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, value: str) -> str:
        if value not in ("rf", "unet"):
            raise ValueError("model_type must be 'rf' or 'unet'")
        return value

    @model_validator(mode="after")
    def validate_model_resolution(self):
        if self.model_type == "unet" and self.resolution != 240:
            raise ValueError("The 2D U-Net is trained for native 240x240 resolution only")
        return self


def dataset_root() -> Path:
    configured = os.getenv("QMI_DATASET_ROOT")
    path = Path(configured).expanduser() if configured else PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour"
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def model_path(resolution: int, model_type: str = "rf") -> Path:
    if model_type == "unet":
        return PROJECT_ROOT / "artifacts" / "msd_unet_240.pt"
    return PROJECT_ROOT / "artifacts" / f"msd_rf_{resolution}.joblib"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "Project Quantum V3.4 API", "version": app.version}


@app.get("/api/dataset/status")
def dataset_status() -> dict[str, Any]:
    root = dataset_root()
    if not root.is_dir():
        return {"available": False, "dataset_root": str(root), "message": "Dataset directory not found"}
    try:
        metadata = read_msd_metadata(root)
        cases = discover_msd_cases(root)
        test_images = len([p for p in (root / "imagesTs").glob("*.nii*") if not p.name.startswith("._")]) if (root / "imagesTs").exists() else 0
        return {"available": True, "dataset_name": "Medical Segmentation Decathlon Task01_BrainTumour",
                "dataset_root": str(root), "labelled_cases": len(cases), "unlabelled_test_images": test_images,
                "modalities": metadata.modalities, "internal_label_map": {"0": "background", "1": "non-enhancing tumour", "2": "edema", "4": "enhancing tumour"},
                "regions": {"WT": [1, 2, 4], "TC": [1, 4], "ET": [4]}}
    except Exception as exc:
        return {"available": False, "dataset_root": str(root), "message": f"Dataset validation failed: {type(exc).__name__}: {exc}"}


@app.get("/api/dataset/cases")
def dataset_cases() -> dict[str, Any]:
    try:
        cases = discover_msd_cases(dataset_root())
        return {"count": len(cases), "cases": [case.case_id for case in cases]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models")
def models() -> dict[str, Any]:
    items = []
    for resolution in (16, 32, 240):
        path = model_path(resolution)
        meta_path = path.with_suffix(".json")
        if path.is_file():
            metadata = {}
            if meta_path.is_file():
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {"metadata_warning": "Model metadata sidecar could not be read"}
            items.append({"name": f"MSD Random Forest {resolution}x{resolution}",
                          "model_type": "rf", "resolution": resolution,
                          "path": str(path), "metadata": metadata})
    unet_path = model_path(240, "unet")
    if unet_path.is_file():
        metadata = {}
        meta_path = unet_path.with_suffix(".json")
        if meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {"metadata_warning": "U-Net metadata sidecar could not be read"}
        items.append({"name": "MSD Residual 2D U-Net 240x240", "model_type": "unet",
                      "resolution": 240, "path": str(unet_path), "metadata": metadata})
    return {"count": len(items), "models": items}


@app.get("/api/experiments")
def experiments(limit: int = 50) -> dict[str, Any]:
    return {"count": min(max(limit, 1), 200), "items": list_experiments(limit=min(max(limit, 1), 200))}


@app.get("/api/predictions")
def predictions(limit: int = 50) -> dict[str, Any]:
    return {"count": min(max(limit, 1), 200), "items": list_predictions(limit=min(max(limit, 1), 200))}


@app.post("/api/experiments", status_code=201)
def record_experiment(payload: ExperimentRecord) -> dict[str, Any]:
    record_id = add_experiment(payload.experiment_name, payload.dataset_name, status=payload.status,
        config=payload.config, metrics=payload.metrics, artifact_dir=payload.artifact_dir)
    return {"id": record_id, "status": "recorded"}


@app.post("/api/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    if payload.resolution not in (16, 32, 240):
        raise HTTPException(status_code=422, detail="resolution must be 16, 32 or native 240")
    path = model_path(payload.resolution, payload.model_type)
    if not path.is_file():
        hint = "scripts/train_unet_msd.py" if payload.model_type == "unet" else "scripts/train_msd.py"
        raise HTTPException(status_code=404, detail=f"Trained {payload.model_type} model not found: {path}. Run {hint} first.")
    try:
        metadata = read_msd_metadata(dataset_root())
        cases = discover_msd_cases(dataset_root())
        case = next((item for item in cases if item.case_id == payload.case_id), None)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Case not found: {payload.case_id}")
        modalities, labels = load_msd_case(case, metadata)
        if payload.slice_index >= labels.shape[0]:
            raise HTTPException(status_code=422, detail=f"slice_index must be below {labels.shape[0]}")
        x, y = resize_slice(modalities[:, payload.slice_index], labels[payload.slice_index], payload.resolution)
        if payload.model_type == "unet":
            try:
                from qmi.unet import predict_unet_slice
                predicted = predict_unet_slice(x, path)
            except ImportError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            meta_path = path.with_suffix(".json")
            model_metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {"model_family": "residual 2D U-Net"}
            model_label = "MSD 2D U-Net 240"
        else:
            model, model_metadata = load_model(path)
            predicted = predict_slice(model, x)
            model_label = f"MSD RF {payload.resolution}"
        scores = segmentation_metrics(predicted, y)
        split_path = PROJECT_ROOT / "outputs" / "reports" / "msd_patient_split.json"
        partition = "unassigned"
        if split_path.is_file():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            for name in ("train", "validation", "test"):
                if payload.case_id in split.get(name, []):
                    partition = name
                    break
        event_id = add_prediction_event(payload.case_id, payload.slice_index, payload.resolution,
            model_label, {"macro_dice": scores["macro_dice"], "macro_iou": scores["macro_iou"]})
        return {"event_id": event_id, "case_id": payload.case_id, "slice_index": payload.slice_index,
                "resolution": payload.resolution, "model_type": payload.model_type,
                "model_name": model_label, "partition": partition,
                "flair": x[0].tolist(), "ground_truth": y.tolist(), "prediction": predicted.tolist(),
                "metrics": json_safe(scores), "model_metadata": model_metadata,
                "note": "Research evaluation on labelled data; not a clinical diagnosis."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {type(exc).__name__}: {exc}") from exc
