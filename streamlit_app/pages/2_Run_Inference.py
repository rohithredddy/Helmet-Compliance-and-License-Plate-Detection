"""
Run Inference page.
Upload images or videos for real-time helmet detection.
Images are automatically resized to 960x960 to match the training process.
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from config.settings import Settings

st.set_page_config(page_title="Run Inference", layout="wide")
st.markdown("## Run Inference")
st.markdown("Upload images or videos to detect helmet violations. Images will be automatically resized to 960x960 to match training conditions for best results.")

# Model loading (cached)
@st.cache_resource
def load_detector():
    from inference.detector import HelmetDetector
    from config.settings import Settings
    
    if not Settings.DEFAULT_MODEL_PATH.exists():
        st.info("Downloading model weights from Snowflake...")
        try:
            from training.model_registry import ModelRegistry
            registry = ModelRegistry()
            download_dir = str(Settings.DEFAULT_MODEL_PATH.parent)
            registry.download_active_model(download_dir)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            
    return HelmetDetector()

try:
    detector = load_detector()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Ensure model weights exist at: results/weights/best.pt")
    st.stop()

st.divider()

# Settings sidebar
with st.sidebar:
    st.markdown("### Inference Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.95, 0.25, 0.05)
    log_to_snowflake = st.checkbox("Log results to Snowflake", value=True)

# File upload
tab_image, tab_video, tab_batch = st.tabs(["Image Upload", "Video Upload", "Batch Processing"])

with tab_image:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if uploaded_file:
        # Convert to numpy array
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        col_orig, col_result = st.columns(2)

        with col_orig:
            st.markdown("#### Original Image")
            st.image(image, use_container_width=True)

        # Run inference
        with st.spinner("Running detection..."):
            result = detector.detect_image(img_array, conf_threshold=conf_threshold)

        with col_result:
            st.markdown("#### Detection Result")
            annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

        # Results summary
        st.divider()
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Detections", result["num_detections"])
        col_m2.metric("Violations", result["num_violations"])
        col_m3.metric("Processing Time", f"{result['processing_time_ms']:.1f} ms")

        # Detection details
        if result["detections"]:
            st.markdown("#### Detection Details")
            det_data = []
            for det in result["detections"]:
                det_data.append({
                    "Class": det["class_name"],
                    "Confidence": f"{det['confidence']:.3f}",
                    "BBox": f"({det['bbox']['x1']}, {det['bbox']['y1']}) - ({det['bbox']['x2']}, {det['bbox']['y2']})",
                })
            st.dataframe(det_data, use_container_width=True, hide_index=True)

        # Violation alerts
        if result["violations"]:
            st.markdown("#### Violation Alerts")
            for v in result["violations"]:
                severity_color = {"high": "orange", "critical": "red"}.get(v["severity"], "yellow")
                st.warning(f"**{v['type'].upper()}** - {v['description']} (Severity: {v['severity']})")

        # Log to Snowflake
        if log_to_snowflake:
            try:
                from inference.result_logger import ResultLogger
                logger = ResultLogger()
                inf_id = logger.log_inference(
                    image_name=uploaded_file.name,
                    result=result,
                    source_type="upload",
                )
                st.caption(f"Logged to Snowflake (ID: {inf_id})")
            except Exception as e:
                st.caption(f"Snowflake logging failed: {e}")

with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded_video:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown("#### Original Video")
            st.video(uploaded_video)

        with col_v2:
            st.markdown("#### Processing Settings")
            frame_skip = st.number_input("Process every N-th frame", min_value=1, max_value=30, value=5)
            max_frames = st.number_input("Max frames to process (0 = all)", min_value=0, value=100)

            if st.button("Run Video Inference"):
                output_path = tmp_path.replace(".mp4", "_output.mp4")
                with st.spinner(f"Processing video (frame_skip={frame_skip})..."):
                    frame_results = detector.detect_video(
                        tmp_path,
                        output_path=output_path,
                        conf_threshold=conf_threshold,
                        frame_skip=frame_skip,
                        max_frames=max_frames if max_frames > 0 else None,
                    )

                st.success(f"Processed {len(frame_results)} frames.")

                total_dets = sum(r["num_detections"] for r in frame_results)
                total_viols = sum(r["num_violations"] for r in frame_results)

                col_vm1, col_vm2, col_vm3 = st.columns(3)
                col_vm1.metric("Frames Processed", len(frame_results))
                col_vm2.metric("Total Detections", total_dets)
                col_vm3.metric("Total Violations", total_viols)

                # Log to Snowflake
                if log_to_snowflake:
                    try:
                        from inference.result_logger import ResultLogger
                        result_logger = ResultLogger()
                        summary = {
                            "num_detections": total_dets,
                            "num_violations": total_viols,
                            "processing_time_ms": sum(r["processing_time_ms"] for r in frame_results),
                            "detections": [],
                            "violations": [],
                        }
                        result_logger.log_inference(
                            image_name=uploaded_video.name,
                            result=summary,
                            source_type="video",
                        )
                        st.caption("Video results logged to Snowflake.")
                    except Exception as e:
                        st.caption(f"Snowflake logging failed: {e}")

with tab_batch:
    st.markdown("#### Batch Processing")
    st.markdown("Process all images in a local directory.")

    input_dir = st.text_input("Input directory path", placeholder="e.g., Inference Data/real imgs")
    output_dir = st.text_input("Output directory path (optional)", placeholder="e.g., batch_output")

    if st.button("Run Batch Inference") and input_dir:
        with st.spinner("Running batch inference..."):
            try:
                from inference.batch_inference import run_batch_inference
                results = run_batch_inference(
                    input_dir=input_dir,
                    output_dir=output_dir if output_dir else None,
                    conf_threshold=conf_threshold,
                    log_to_snowflake=log_to_snowflake,
                )
                st.success(f"Batch complete. Processed {len(results)} files.")

                result_data = []
                for r in results:
                    result_data.append({
                        "File": r["filename"],
                        "Type": r["type"],
                        "Detections": r.get("num_detections", "-"),
                        "Violations": r.get("num_violations", "-"),
                    })
                st.dataframe(result_data, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Batch inference failed: {e}")
