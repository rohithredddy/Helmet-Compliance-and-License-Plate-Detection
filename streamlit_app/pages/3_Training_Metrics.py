"""
Training Metrics page.
Displays training loss curves, validation metrics, and model registry.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from components.snowflake_auth import require_snowflake
from components.charts import (
    training_loss_curves,
    validation_metrics_chart,
)
from config.settings import Settings

st.set_page_config(page_title="Training Metrics", layout="wide")
st.markdown("## Training Metrics")
st.markdown("View training history, loss curves, and model performance.")

sf = require_snowflake()

# Option to upload existing results
with st.expander("Import Historical Training Results"):
    st.markdown("Import metrics from an existing results.csv file into Snowflake.")
    if st.button("Import from results/results.csv"):
        with st.spinner("Importing historical results..."):
            try:
                from training.experiment_tracker import ExperimentTracker
                tracker = ExperimentTracker()
                run_id = tracker.log_existing_results(str(Settings.RESULTS_CSV))
                st.success(f"Imported successfully. Run ID: {run_id}")
            except Exception as e:
                st.error(f"Import failed: {e}")

    if st.button("Register Existing Model (v1.0)"):
        with st.spinner("Registering model..."):
            try:
                from training.model_registry import ModelRegistry
                registry = ModelRegistry()
                version = registry.register_existing_model()
                st.success(f"Registered and promoted: {version}")
            except Exception as e:
                st.error(f"Registration failed: {e}")

st.divider()

# Fetch training runs
try:
    runs_df = sf.fetch_dataframe(
        "SELECT * FROM MODELS.TRAINING_RUNS ORDER BY STARTED_AT DESC",
        schema="MODELS",
    )
except Exception as e:
    runs_df = pd.DataFrame()
    st.warning(f"Could not fetch training runs: {e}")

if runs_df.empty:
    st.info(
        "No training runs found. Use 'Import Historical Training Results' above "
        "to load existing data."
    )
    st.stop()

# Run selector
st.markdown("### Training Runs")

run_options = {
    f"{row['RUN_ID']} ({row['STATUS']})": row['RUN_ID']
    for _, row in runs_df.iterrows()
}
selected_label = st.selectbox("Select training run", list(run_options.keys()))
selected_run_id = run_options[selected_label]

# Run summary
run_info = runs_df[runs_df["RUN_ID"] == selected_run_id].iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Status", run_info.get("STATUS", "-"))
col2.metric("mAP50", f"{run_info.get('BEST_MAP50', 0):.4f}" if run_info.get("BEST_MAP50") else "-")
col3.metric("mAP50-95", f"{run_info.get('BEST_MAP50_95', 0):.4f}" if run_info.get("BEST_MAP50_95") else "-")
col4.metric("Precision", f"{run_info.get('BEST_PRECISION', 0):.4f}" if run_info.get("BEST_PRECISION") else "-")
col5.metric("Recall", f"{run_info.get('BEST_RECALL', 0):.4f}" if run_info.get("BEST_RECALL") else "-")

st.divider()

# Fetch epoch-level metrics
try:
    metrics_df = sf.fetch_dataframe(
        f"SELECT * FROM MODELS.TRAINING_METRICS WHERE RUN_ID = '{selected_run_id}' ORDER BY EPOCH",
        schema="MODELS",
    )
except Exception as e:
    metrics_df = pd.DataFrame()
    st.warning(f"Could not fetch metrics: {e}")

if not metrics_df.empty:
    # Loss curves
    st.markdown("### Training Loss Curves")
    fig_loss = training_loss_curves(metrics_df)
    st.plotly_chart(fig_loss, use_container_width=True)

    # Validation metrics
    st.markdown("### Validation Metrics")
    fig_val = validation_metrics_chart(metrics_df)
    st.plotly_chart(fig_val, use_container_width=True)

    # Raw metrics table
    with st.expander("Raw Metrics Table"):
        display_cols = [
            c for c in metrics_df.columns
            if c not in ("METRIC_ID", "RUN_ID", "RECORDED_AT")
        ]
        st.dataframe(
            metrics_df[display_cols].round(4),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("No epoch-level metrics found for this run.")

st.divider()

# Model Registry
st.markdown("### Model Registry")

try:
    from training.model_registry import ModelRegistry
    registry = ModelRegistry()
    models_df = registry.list_models()

    if not models_df.empty:
        st.dataframe(models_df, use_container_width=True, hide_index=True)

        # Promote model
        model_versions = models_df["MODEL_VERSION"].tolist()
        promote_version = st.selectbox("Select model to promote", model_versions)
        if st.button("Promote Selected Model"):
            registry.promote_model(promote_version)
            st.success(f"Promoted model: {promote_version}")
            st.rerun()
    else:
        st.info("No models registered yet.")
except Exception as e:
    st.warning(f"Could not access model registry: {e}")
