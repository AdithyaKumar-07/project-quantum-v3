from __future__ import annotations
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Project Quantum V3.4", page_icon="⚛️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2.2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: linear-gradient(135deg, rgba(45,75,130,.12), rgba(90,130,190,.04)); border: 1px solid rgba(120,145,180,.25); padding: 14px 16px; border-radius: 14px;}
.hero {padding: 1.45rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#111d36,#1c3b65 70%,#255a78); color: #f3f7ff; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: -.035em;}
.hero p {margin: .45rem 0 0 0; color: #c9d8ef; font-size: 1rem;}
.small-note {color: #8c98a8; font-size: .82rem;}
</style>
<div class="hero"><h1>⚛️ CAPSTONE PROJECT </h1><p>Hybrid Quantum–Classical Medical Image Processing · FRQI / NEQR · Brain Tumour Segmentation</p></div>
""", unsafe_allow_html=True)

API_URL = st.sidebar.text_input("Backend API", os.getenv("QMI_API_URL", "http://127.0.0.1:8000")).rstrip("/")
page = st.sidebar.radio("Workspace", ["Overview", "Dataset & Inference", "Model Evaluation", "Quantum Lab", "Experiment History"])


def api_get(path: str, params: dict | None = None):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def figure_images(items, height=320):
    fig = make_subplots(rows=1, cols=len(items), subplot_titles=[x[0] for x in items], horizontal_spacing=0.035)
    for idx, (_, arr, cmap) in enumerate(items, start=1):
        fig.add_trace(go.Heatmap(z=np.asarray(arr), colorscale=cmap, showscale=False), row=1, col=idx)
        fig.update_xaxes(visible=False, row=1, col=idx); fig.update_yaxes(visible=False, autorange="reversed", row=1, col=idx)
    fig.update_layout(height=height, margin=dict(l=2,r=2,t=35,b=2), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def fmt_metric(value, digits=4):
    """Render undefined/non-finite metrics without crashing the Streamlit view."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if np.isfinite(number) else "—"


if page == "Overview":
    st.subheader("Research workspace")
    try:
        health = api_get("/api/health")
        status = api_get("/api/dataset/status")
        models = api_get("/api/models")
        runs = api_get("/api/experiments", {"limit": 100})
        api_ok = True
    except Exception as exc:
        api_ok = False
        st.error(f"Backend is not reachable: {exc}")
        st.code("run_api.bat", language="bat")
        health, status, models, runs = {}, {}, {"count": 0, "models": []}, {"items": []}
    a,b,c,d = st.columns(4)
    a.metric("API", "Online" if api_ok else "Offline")
    b.metric("Labelled patients", status.get("labelled_cases", "—") if status.get("available") else "Not found")
    c.metric("Trained models", models.get("count", 0))
    d.metric("Experiments logged", len(runs.get("items", [])))
    st.markdown("### Project status")
    if status.get("available"):
        st.success("MSD Task01_BrainTumour dataset structure validated by the backend.")
        st.write(f"**Dataset path:** `{status['dataset_root']}`")
        st.write("**Modalities:** FLAIR · T1 · T1ce · T2")
        st.write("**Regions:** WT = {1,2,4}; TC = {1,4}; ET = {4}")
    else:
        st.warning(status.get("message", "Dataset not yet available. Add the labelled Task01_BrainTumour dataset to data/raw/Task01_BrainTumour."))
    st.markdown("### Research workflow")
    steps = [
        ("01 · Dataset", "MSD 4D NIfTI reader, channel verification and patient-level split"),
        ("02 · Classical models", "Random Forest baselines plus the optional native-resolution residual 2D U-Net"),
        ("03 · Quantum encodings", "FRQI/NEQR at 16×16 and 32×32 plus 64×64, 128×128 and native 240×240 patchwise reconstruction"),
        ("04 · Hybrid evaluation", "Quantum FLAIR reconstruction followed by classical segmentation"),
        ("05 · Reporting", "Dice, IoU, precision, recall, specificity, HD95, MSE, PSNR and SSIM"),
    ]
    for title, desc in steps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)
    st.info("Research software only. Not a clinical diagnostic system. Synthetic demo artifacts are not medical-dataset results.")

elif page == "Dataset & Inference":
    st.subheader("Dataset explorer and trained-model inference")
    try:
        status = api_get("/api/dataset/status")
        if not status.get("available"):
            st.warning(status.get("message", "Dataset not available")); st.stop()
        cases_data = api_get("/api/dataset/cases")
        model_data = api_get("/api/models")
    except Exception as exc:
        st.error(f"Could not retrieve dataset metadata: {exc}"); st.stop()
    if not cases_data["cases"]:
        st.warning("No labelled cases were found."); st.stop()
    case_id = st.selectbox("Patient case", cases_data["cases"])
    model_choice = st.selectbox("Segmentation model", [
        "Random Forest (FRQI/NEQR-compatible)", "Residual 2D U-Net (native 240×240)"])
    model_type = "unet" if model_choice.startswith("Residual") else "rf"
    col_a, col_b = st.columns(2)
    if model_type == "unet":
        resolution = 240
        col_a.info("Native 240×240 · 2D U-Net")
    else:
        resolution = col_a.selectbox("Model resolution", [16, 32, 240], index=1,
            format_func=lambda x: "Native 240×240" if x == 240 else f"{x}×{x}")
    slice_idx = col_b.number_input("Axial slice index", min_value=0, value=70, step=1)
    matching_model = any(x.get("model_type", "rf") == model_type and x["resolution"] == resolution
                         for x in model_data["models"])
    if not matching_model:
        hint = "python scripts/train_unet_msd.py" if model_type == "unet" else "python scripts/train_msd.py"
        st.warning(f"No trained {model_choice} checkpoint is available. Run `{hint}` first.")
    if st.button("Run segmentation inference", type="primary", disabled=not matching_model):
        with st.spinner("Loading NIfTI case and running segmentation..."):
            try:
                result = api_post("/api/predict", {"case_id": case_id, "slice_index": int(slice_idx),
                    "resolution": int(resolution), "model_type": model_type})
                st.session_state["last_prediction"] = result
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Inference request failed: {detail}")
            except Exception as exc:
                st.error(f"Inference failed: {type(exc).__name__}: {exc}")
    result = st.session_state.get("last_prediction")
    if result and result.get("case_id") == case_id and result.get("resolution") == resolution and result.get("model_type", "rf") == model_type:
        st.markdown(f"#### {result['case_id']} · axial slice {result['slice_index']}")
        calibration_info = (result.get("model_metadata") or {}).get("probability_calibration") or {}
        calibrated_active = bool(calibration_info.get("accepted") and calibration_info.get("tumour_probability_threshold") is not None)
        if model_type == "unet":
            prediction_title = "Residual 2D U-Net prediction"
            st.caption("Native-resolution model selected using validation patients; held-out test results are reported separately.")
        else:
            prediction_title = "Validation-calibrated RF" if calibrated_active else "RF prediction (raw argmax)"
        image_panels = []
        if result.get("flair") is not None:
            image_panels.append(("Input FLAIR", result["flair"], "Gray"))
        image_panels.extend([("Ground truth", result["ground_truth"], "Viridis"),
                             (prediction_title, result["prediction"], "Viridis")])
        st.plotly_chart(figure_images(image_panels), use_container_width=True)
        if calibration_info:
            if calibrated_active:
                st.caption(f"Validation-selected tumour probability threshold: {float(calibration_info['tumour_probability_threshold']):.3f}")
            else:
                st.caption("Validation calibration did not outperform raw argmax; original model decision rule retained.")
        metrics = result["metrics"]
        pred_array = np.asarray(result["prediction"], dtype=np.uint8)
        gt_array = np.asarray(result["ground_truth"], dtype=np.uint8)
        pred_coverage = float(np.mean(pred_array != 0))
        gt_coverage = float(np.mean(gt_array != 0))
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Active-region Macro Dice", fmt_metric(metrics.get("macro_dice_active")))
        m2.metric("Active-region Macro IoU", fmt_metric(metrics.get("macro_iou_active")))
        m3.metric("WT Dice", fmt_metric(metrics["WT"]["dice"]))
        m4.metric("ET Dice", fmt_metric(metrics["ET"]["dice"]))
        st.caption(f"Patient partition: {result.get('partition', 'unassigned')}. Regions with tumour in ground truth: {metrics.get('n_regions_present_gt', '—')} / 3. Empty/empty regions are reported separately and do not enter active-region macro scores. Predicted tumour coverage: {pred_coverage:.3%}; ground-truth coverage: {gt_coverage:.3%}.")
        if pred_coverage > max(0.05, 5.0 * max(gt_coverage, 1e-6)):
            st.warning("Potential over-segmentation: predicted tumour area is more than five times the ground-truth area on this labelled slice. Review precision and the prediction overlay.")
        if result.get("partition") != "test":
            st.warning("This patient is not in the held-out test partition. Treat these scores as exploratory, not final test results.")
        rows = []
        for region in ("WT", "TC", "ET"):
            rows.append({"Region": region, **{k: metrics[region].get(k) for k in ("dice","iou","precision","recall","specificity","hd95_pixels")},
                         "Empty GT": metrics[region].get("empty_ground_truth"),
                         "Empty prediction": metrics[region].get("empty_prediction")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        out = io.BytesIO(); np.savez_compressed(out, ground_truth=np.asarray(result["ground_truth"], dtype=np.uint8), prediction=np.asarray(result["prediction"], dtype=np.uint8))
        st.download_button("Download prediction + ground truth", out.getvalue(), f"{case_id}_slice{slice_idx}_segmentation.npz", "application/octet-stream")
        st.caption(result["note"])

elif page == "Model Evaluation":
    st.subheader("Held-out evaluation reports")
    reports = ROOT / "outputs" / "reports"
    family = st.selectbox("Model family", ["Random Forest", "Residual 2D U-Net"])
    if family == "Residual 2D U-Net":
        resolution = 240
        calibrated_test_csv = reports / "msd_unet_test_summary.csv"
        calibrated_val_csv = reports / "msd_unet_validation_summary.csv"
        test_csv, val_csv = calibrated_test_csv, calibrated_val_csv
        missing_command = "python scripts/train_unet_msd.py"
        report_status = "Residual 2D U-Net; best checkpoint selected using validation patients."
    else:
        resolution = st.selectbox("Resolution", [16, 32, 240], index=1,
            format_func=lambda x: "Native 240×240" if x == 240 else f"{x}×{x}")
        calibrated_test_csv = reports / f"msd_test_calibrated_summary_{resolution}.csv"
        calibrated_val_csv = reports / f"msd_validation_calibrated_summary_{resolution}.csv"
        test_csv = calibrated_test_csv if calibrated_test_csv.is_file() else reports / f"msd_test_summary_{resolution}.csv"
        val_csv = calibrated_val_csv if calibrated_val_csv.is_file() else reports / f"msd_validation_summary_{resolution}.csv"
        missing_command = "python scripts/train_msd.py --max-patients 0 --resolution all"
        report_status = ("Validation-calibrated held-out RF reports" if calibrated_test_csv.is_file()
                         else "Uncalibrated RF reports; run scripts/optimize_msd_models.py to fit thresholds on validation patients.")
    if not test_csv.exists():
        st.info("No held-out report for this model has been generated yet.")
        st.code(missing_command, language="powershell")
    else:
        df = pd.read_csv(test_csv)
        st.caption(report_status + " Verify the patient split JSON before publication.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        if "macro_dice_active_mean" in df.columns:
            chart = df[["method", "macro_dice_active_mean", "macro_iou_active_mean"]].melt("method", var_name="Metric", value_name="Score")
            chart["Metric"] = chart["Metric"].replace({"macro_dice_active_mean": "Active-region Macro Dice", "macro_iou_active_mean": "Active-region Macro IoU"})
            st.plotly_chart(px.bar(chart, x="method", y="Score", color="Metric", barmode="group", title="Held-out active-region test performance"), use_container_width=True)
        st.download_button("Download held-out report CSV", test_csv.read_bytes(), test_csv.name, "text/csv")
        if family == "Residual 2D U-Net":
            paired_csv = reports / "msd_unet_vs_rf_paired_test_summary.csv"
            if paired_csv.is_file():
                st.markdown("#### Paired comparison on identical held-out slices")
                st.caption("Only generated when the RF and U-Net artifacts record the same patient-level split. Both models are evaluated on the same source axial slices.")
                st.dataframe(pd.read_csv(paired_csv), use_container_width=True, hide_index=True)
                st.download_button("Download paired RF vs U-Net CSV", paired_csv.read_bytes(), paired_csv.name, "text/csv")
    if val_csv.exists():
        with st.expander("Validation report"):
            st.dataframe(pd.read_csv(val_csv), use_container_width=True, hide_index=True)

elif page == "Quantum Lab":
    st.subheader("FRQI / NEQR quantum and hybrid laboratory")
    st.info("16×16 and 32×32 use the direct Aer workflow. 64×64, 128×128 and native 240×240 use complete 32×32 patch-wise exact state-vector reconstruction, with optional bounded Aer spot-checks on representative patches.")
    try:
        from qmi.data.msd_dataset import read_msd_metadata, discover_msd_cases, load_msd_case
        from qmi.data.preprocessing import resize_slice
        from qmi.models import load_model
        from qmi.hybrid import run_hybrid_path
        from qmi.quantum.patchwise import run_patchwise_hybrid
        import cv2
    except Exception as exc:
        st.error(f"Quantum components could not be imported: {exc}"); st.stop()
    try:
        status = api_get("/api/dataset/status")
        cases_data = api_get("/api/dataset/cases")
    except Exception as exc:
        st.error(f"Backend not available: {exc}"); st.stop()
    if not status.get("available"):
        st.warning("Add the labelled MSD dataset first."); st.stop()
    split_path = ROOT / "outputs" / "reports" / "msd_patient_split.json"
    if not split_path.is_file():
        st.warning("Run the patient-level training workflow first; the saved held-out test split is required for quantum comparison.")
        st.stop()
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    heldout_cases = [case_id for case_id in split_data.get("test", []) if case_id in set(cases_data["cases"])]
    if not heldout_cases:
        st.error("No cases from the saved test partition are available.")
        st.stop()
    qcase_id = st.selectbox("Held-out test patient", heldout_cases, key="qcase")
    q1, q2, q3 = st.columns(3)
    qresolution = q1.selectbox("Quantum image resolution", [16, 32, 64, 128, 240], index=4, key="qresolution",
        format_func=lambda x: "Native 240×240" if x == 240 else f"{x}×{x}")
    representation = q2.selectbox("Encoding", ["FRQI", "NEQR"], key="qrepresentation")
    highres = qresolution >= 64
    if not highres:
        qmode = q3.selectbox("Aer simulation", ["ideal", "noisy"], key="qmode_direct")
    else:
        highres_modes = {
            "Exact full-image patchwise statevector": "statevector",
            "Aer ideal FRQI on every tile": "aer_ideal_full",
            "Aer ideal spot-check + exact full image": "aer_ideal_spotcheck",
            "Aer noisy spot-check + exact full image": "aer_noisy_spotcheck",
        }
        if representation == "NEQR":
            highres_modes.pop("Aer ideal FRQI on every tile")
        selected_mode_label = q3.selectbox("High-resolution workflow", list(highres_modes), key=f"qmode_patchwise_{representation}")
        qmode = highres_modes[selected_mode_label]
    qslice = st.number_input("Axial slice", min_value=0, value=70, step=1, key="qslice")
    max_aer_tiles = 1
    allow_expensive = False
    if highres and qmode in {"aer_ideal_spotcheck", "aer_noisy_spotcheck"}:
        c1, c2 = st.columns(2)
        max_aer_tiles = int(c1.number_input("Aer spot-check patches (1–4)", min_value=1, max_value=4, value=1, step=1))
        if representation == "NEQR":
            allow_expensive = c2.checkbox("I understand 32×32 NEQR Aer is an 18-qubit CPU workload", value=False)
            st.warning("NEQR exact patchwise reconstruction covers the full image. Aer spot-checks are separate; an 18-qubit NEQR Aer run can be very slow and requires explicit opt-in.")
        else:
            c2.caption("FRQI Aer spot-checks run on representative 32×32 tiles only.")
    if highres:
        st.caption(f"Patch plan: 32×32 tiles · {((qresolution + 31)//32)**2} tiles · full-image reconstruction plus native 240×240 RF inference.")
        if qmode == "aer_ideal_full":
            st.warning("This runs one Qiskit Aer ideal FRQI circuit for every image tile. The 240×240 image requires 64 sequential Aer circuit runs.")
    run_disabled = bool(highres and qmode != "statevector" and representation == "NEQR" and not allow_expensive)
    if not highres and qresolution == 32 and representation == "NEQR":
        st.warning("32×32 NEQR Aer is an 18-qubit workload. Use the high-resolution exact patchwise statevector path for a complete image, or explicitly benchmark one Aer tile from the CLI.")
        run_disabled = True
    if st.button("Run quantum reconstruction + hybrid segmentation", type="primary", disabled=run_disabled):
        model_resolution = 240 if highres else qresolution
        model_file = ROOT / "artifacts" / f"msd_rf_{model_resolution}.joblib"
        if not model_file.exists():
            st.error(f"Trained model missing: {model_file}. Train/calibrate the matching model first.")
        else:
            try:
                md = read_msd_metadata(status["dataset_root"])
                case = next(c for c in discover_msd_cases(status["dataset_root"]) if c.case_id == qcase_id)
                volumes, masks = load_msd_case(case, md)
                if qslice >= masks.shape[0]:
                    st.error(f"Slice index must be below {masks.shape[0]}.")
                elif not np.any(masks[int(qslice)] > 0):
                    st.warning("This slice has no tumour in the ground truth. Select a tumour-positive axial slice before running the hybrid comparison.")
                else:
                    native_modalities = volumes[:, int(qslice)]
                    native_truth = masks[int(qslice)]
                    model, model_metadata = load_model(model_file)
                    with st.spinner("Preparing quantum patches, reconstructing FLAIR and running native-resolution RF inference..."):
                        if not highres:
                            m, y = resize_slice(native_modalities, native_truth, int(qresolution))
                            quantum_result = run_hybrid_path(m, model, representation, true_labels=y,
                                simulation_mode=qmode, shots=max(20000, qresolution*qresolution*100) if qmode == "noisy" else None,
                                device="CPU")
                            original_image = m[0]
                            reconstructed_image = quantum_result.reconstructed_flair
                            prediction_image = quantum_result.predicted_labels
                            metrics_payload = {"reconstruction": quantum_result.reconstruction,
                                "segmentation": quantum_result.segmentation,
                                "simulation_seconds": quantum_result.simulation_seconds,
                                "inference_seconds": quantum_result.inference_seconds,
                                "simulator": quantum_result.simulator_metadata,
                                "model_metadata": model_metadata}
                            download_arrays = {"reconstructed_flair": quantum_result.reconstructed_flair,
                                "prediction": quantum_result.predicted_labels, "ground_truth": y}
                            download_name = f"{qcase_id}_{representation.lower()}_{qresolution}_{qmode}.npz"
                        else:
                            quantum_result = run_patchwise_hybrid(native_modalities, model, representation,
                                true_labels=native_truth, quantum_resolution=int(qresolution),
                                mode=qmode, max_aer_tiles=max_aer_tiles, shots=50000,
                                allow_expensive_18_qubit=allow_expensive)
                            if qresolution == 240:
                                original_image = np.rint(native_modalities[0] * 255).clip(0,255).astype(np.uint8)
                            else:
                                original_image = cv2.resize(native_modalities[0], (qresolution, qresolution),
                                    interpolation=cv2.INTER_AREA).astype(np.float32)
                            reconstructed_image = quantum_result.reconstructed_flair
                            prediction_image = quantum_result.predicted_labels_native
                            metrics_payload = {"reconstruction_at_quantum_resolution": quantum_result.reconstruction,
                                "native_240_segmentation": quantum_result.segmentation,
                                "workflow_metadata": quantum_result.metadata,
                                "patch_records": quantum_result.patch_records,
                                "aer_tile_results": quantum_result.aer_tile_records,
                                "total_seconds": quantum_result.total_seconds,
                                "model_metadata": model_metadata}
                            download_arrays = {"original_flair_native": native_modalities[0],
                                "reconstructed_flair_quantum_resolution": quantum_result.reconstructed_flair,
                                "reconstructed_flair_native": quantum_result.reconstructed_flair_native,
                                "prediction_native": quantum_result.predicted_labels_native,
                                "ground_truth_native": native_truth}
                            download_name = f"{qcase_id}_{representation.lower()}_{qresolution}_patchwise_{qmode}.npz"
                    st.plotly_chart(figure_images([
                        (f"Original FLAIR · {qresolution}×{qresolution}", original_image, "Greys"),
                        ("Quantum reconstruction", reconstructed_image, "Greys"),
                        ("RF segmentation · native 240×240" if highres else "Hybrid segmentation", prediction_image, "Viridis")
                    ], height=390), use_container_width=True)
                    if highres:
                        if qmode == "aer_ideal_full":
                            st.success(f"Full image reconstructed from {quantum_result.metadata['aer_full_image_tile_runs']} Qiskit Aer ideal FRQI tile runs.")
                        else:
                            st.success(f"Processed the complete image in {quantum_result.metadata['total_patches']} patches. Full reconstruction used exact state-vector encode/decode; any Aer spot-checks are reported separately.")
                    st.json(metrics_payload)
                    out = io.BytesIO(); np.savez_compressed(out, **download_arrays)
                    st.download_button("Download quantum + hybrid outputs", out.getvalue(), download_name, "application/octet-stream")
            except Exception as exc:
                st.exception(exc)

elif page == "Experiment History":
    st.subheader("Experiment and prediction history")
    try:
        runs = api_get("/api/experiments", {"limit": 100})["items"]
        predictions = api_get("/api/predictions", {"limit": 100})["items"]
    except Exception as exc:
        st.error(f"Could not load history: {exc}"); st.stop()
    st.markdown("### Model training runs")
    if runs:
        st.dataframe(pd.DataFrame([{k: item.get(k) for k in ("id","experiment_name","dataset_name","status","created_at") } for item in runs]), use_container_width=True, hide_index=True)
    else:
        st.info("No experiment runs have been recorded yet.")
    st.markdown("### Inference events")
    if predictions:
        st.dataframe(pd.DataFrame([{k: item.get(k) for k in ("id","case_id","slice_index","resolution","model_name","created_at") } for item in predictions]), use_container_width=True, hide_index=True)
    else:
        st.info("No inference events have been recorded yet.")

st.sidebar.divider()
st.sidebar.caption("Project Quantum V3 · Research prototype · Not for clinical use")
