"""
Retrain Model page.
Upload new data, trigger retraining, and monitor pipeline progress.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from components.snowflake_auth import require_snowflake
from config.settings import Settings

st.set_page_config(page_title="Retrain Model", layout="wide")
st.markdown("## Retrain Model")
st.markdown("Upload new training data and trigger the retraining pipeline.")

sf = require_snowflake()

st.divider()

# --- Section 1: Add New Data to Queue ---
st.markdown("### Add New Data to Retraining Queue")

col_upload, col_info = st.columns([2, 1])

with col_upload:
    batch_name = st.text_input("Batch name", placeholder="e.g., april_2026_night_images")
    new_data_dir = st.text_input(
        "Path to new data directory",
        placeholder="e.g., C:/path/to/new_images",
    )

    if st.button("Add to Queue") and batch_name and new_data_dir:
        data_path = Path(new_data_dir)
        if not data_path.exists():
            st.error(f"Directory not found: {new_data_dir}")
        else:
            # Count images
            image_count = sum(
                1 for f in data_path.rglob("*")
                if f.suffix.lower() in Settings.IMAGE_EXTENSIONS
            )

            if image_count == 0:
                st.warning("No images found in the specified directory.")
            else:
                try:
                    from retraining.retrain_trigger import RetrainTrigger
                    trigger = RetrainTrigger()
                    trigger.add_to_queue(
                        batch_name=batch_name,
                        num_images=image_count,
                        stage_path=str(data_path),
                        requested_by="streamlit_user",
                    )
                    st.success(
                        f"Added batch '{batch_name}' with {image_count} images to queue."
                    )
                except Exception as e:
                    st.error(f"Failed to add to queue: {e}")

with col_info:
    st.markdown("#### Requirements")
    st.markdown(f"""
    - Minimum **{Settings.MIN_RETRAIN_IMAGES}** new images required
    - Images should be in YOLO format
    - Labels in `.txt` files (same directory or `/labels` subfolder)
    - Supported formats: {', '.join(Settings.IMAGE_EXTENSIONS)}
    """)

st.divider()

# --- Section 2: Retraining Queue Status ---
st.markdown("### Retraining Queue")

try:
    queue_df = sf.fetch_dataframe(
        "SELECT * FROM RESULTS.RETRAINING_QUEUE ORDER BY REQUESTED_AT DESC",
        schema="RESULTS",
    )

    if not queue_df.empty:
        st.dataframe(queue_df, use_container_width=True, hide_index=True)

        # Summary
        pending = queue_df[queue_df["STATUS"] == "PENDING"]
        total_pending = int(pending["NUM_IMAGES"].sum()) if not pending.empty else 0
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Pending Batches", len(pending))
        col_s2.metric("Total Pending Images", total_pending)
        col_s3.metric("Min Required", Settings.MIN_RETRAIN_IMAGES)
    else:
        st.info("No batches in the retraining queue.")
except Exception as e:
    st.warning(f"Could not fetch queue: {e}")

st.divider()

# --- Section 3: Trigger Retraining ---
st.markdown("### Trigger Retraining Pipeline")

col_trigger, col_status = st.columns(2)

with col_trigger:
    st.markdown("#### Automatic Trigger")
    st.markdown("Check if conditions are met and run the pipeline.")

    if st.button("Check Trigger Conditions"):
        try:
            from retraining.retrain_trigger import RetrainTrigger
            trigger = RetrainTrigger()
            evaluation = trigger.evaluate_trigger()

            if evaluation["should_retrain"]:
                st.success(evaluation["reason"])
            else:
                st.info(evaluation["reason"])

            st.json(evaluation)
        except Exception as e:
            st.error(f"Trigger check failed: {e}")

    st.markdown("#### Manual Trigger")
    force_retrain = st.checkbox("Force retraining (skip minimum data check)")
    retrain_data_dir = st.text_input(
        "New data directory for retraining",
        placeholder="Leave empty to use existing dataset only",
        key="retrain_dir",
    )

    if st.button("Start Retraining Pipeline"):
        st.warning("This will train a new model. This may take a long time.")
        confirm = st.checkbox("I understand, proceed with retraining")
        if confirm:
            with st.spinner("Running retraining pipeline..."):
                try:
                    from retraining.retrain_pipeline import RetrainPipeline
                    pipeline = RetrainPipeline()
                    result = pipeline.run(
                        new_data_dir=retrain_data_dir if retrain_data_dir else None,
                        force=force_retrain,
                    )
                    st.json(result)

                    if result["status"] == "completed":
                        if result.get("promoted"):
                            st.success("New model trained and promoted as active.")
                        else:
                            st.info("New model trained but existing model is still better.")
                    elif result["status"] == "skipped":
                        st.info(f"Retraining skipped: {result['reason']}")
                    else:
                        st.error(f"Retraining failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Retraining pipeline error: {e}")

with col_status:
    st.markdown("#### Current Active Model")
    try:
        from training.model_registry import ModelRegistry
        registry = ModelRegistry()
        active = registry.get_active_model()

        if active:
            st.metric("Version", active["model_version"])
            st.metric("mAP50", f"{active.get('mAP50', 0):.4f}")
            st.metric("mAP50-95", f"{active.get('mAP50-95', 0):.4f}")
        else:
            st.info("No active model registered.")
    except Exception as e:
        st.warning(f"Could not fetch active model: {e}")

    st.markdown("#### Data Drift Analysis")
    drift_dir = st.text_input(
        "Directory to check for drift",
        placeholder="Path to new data",
        key="drift_dir",
    )
    if st.button("Analyze Drift") and drift_dir:
        try:
            from retraining.data_drift_monitor import DataDriftMonitor
            monitor = DataDriftMonitor()
            drift = monitor.analyze_new_data(drift_dir)
            st.json(drift)

            if drift.get("drift_detected"):
                st.warning(drift["recommendation"])
            else:
                st.success(drift.get("recommendation", "No significant drift."))
        except Exception as e:
            st.error(f"Drift analysis failed: {e}")
