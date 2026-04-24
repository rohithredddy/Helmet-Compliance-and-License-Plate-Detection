"""
Detection History page.
Query and visualize historical detection results from Snowflake.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from components.snowflake_auth import require_snowflake
from components.charts import violation_timeline, detection_confidence_histogram

st.set_page_config(page_title="Detection History", layout="wide")
st.markdown("## Detection History")
st.markdown("Browse and analyze historical inference results and violations.")

sf = require_snowflake()

st.divider()

# --- Filters ---
st.markdown("### Filters")

col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    days_filter = st.selectbox("Time range", [7, 14, 30, 90, 365], index=2)

with col_f2:
    source_filter = st.selectbox("Source type", ["All", "upload", "batch", "video"])

with col_f3:
    violation_filter = st.selectbox(
        "Violation type",
        ["All", "no_helmet", "None (no violation)"],
    )

with col_f4:
    limit = st.number_input("Max results", min_value=10, max_value=1000, value=100)

st.divider()

# --- Inference Logs ---
st.markdown("### Inference Logs")

try:
    source_clause = ""
    if source_filter != "All":
        source_clause = f"AND SOURCE_TYPE = '{source_filter}'"

    logs_df = sf.fetch_dataframe(
        f"""
        SELECT
            INFERENCE_ID, IMAGE_NAME, SOURCE_TYPE, MODEL_VERSION,
            NUM_DETECTIONS, NUM_VIOLATIONS, PROCESSING_TIME_MS, CREATED_AT
        FROM RESULTS.INFERENCE_LOGS
        WHERE CREATED_AT >= DATEADD(day, -{days_filter}, CURRENT_TIMESTAMP())
        {source_clause}
        ORDER BY CREATED_AT DESC
        LIMIT {limit}
        """,
        schema="RESULTS",
    )

    if not logs_df.empty:
        # Summary metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Inferences", len(logs_df))
        col_m2.metric("Total Detections", int(logs_df["NUM_DETECTIONS"].sum()))
        col_m3.metric("Total Violations", int(logs_df["NUM_VIOLATIONS"].sum()))
        avg_time = logs_df["PROCESSING_TIME_MS"].mean()
        col_m4.metric("Avg Processing Time", f"{avg_time:.1f} ms")

        st.dataframe(logs_df, use_container_width=True, hide_index=True)

        # Export
        csv = logs_df.to_csv(index=False)
        st.download_button(
            "Download as CSV",
            data=csv,
            file_name="inference_logs.csv",
            mime="text/csv",
        )
    else:
        st.info("No inference logs found for the selected filters.")

except Exception as e:
    st.warning(f"Could not fetch inference logs: {e}")
    logs_df = pd.DataFrame()

st.divider()

# --- Detection Results ---
st.markdown("### Detection Results")

try:
    violation_clause = ""
    if violation_filter == "no_helmet":
        violation_clause = "AND d.VIOLATION_TYPE = 'no_helmet'"
    elif violation_filter == "None (no violation)":
        violation_clause = "AND d.VIOLATION_TYPE IS NULL"

    detections_df = sf.fetch_dataframe(
        f"""
        SELECT
            d.DETECTION_ID, d.INFERENCE_ID, d.CLASS_NAME, d.CLASS_ID,
            d.CONFIDENCE, d.BBOX_X1, d.BBOX_Y1, d.BBOX_X2, d.BBOX_Y2,
            d.VIOLATION_TYPE, d.CREATED_AT
        FROM RESULTS.DETECTION_RESULTS d
        JOIN RESULTS.INFERENCE_LOGS i ON d.INFERENCE_ID = i.INFERENCE_ID
        WHERE i.CREATED_AT >= DATEADD(day, -{days_filter}, CURRENT_TIMESTAMP())
        {violation_clause}
        ORDER BY d.CREATED_AT DESC
        LIMIT {limit}
        """,
        schema="RESULTS",
    )

    if not detections_df.empty:
        # Class breakdown
        st.markdown("#### Detection Class Breakdown")
        class_counts = detections_df["CLASS_NAME"].value_counts().to_dict()

        class_cols = st.columns(len(class_counts))
        for i, (class_name, count) in enumerate(class_counts.items()):
            class_cols[i].metric(class_name, count)

        # Confidence histogram
        fig_conf = detection_confidence_histogram(detections_df)
        st.plotly_chart(fig_conf, use_container_width=True)

        # Violations breakdown
        violations_only = detections_df[detections_df["VIOLATION_TYPE"].notna()]
        if not violations_only.empty:
            st.markdown("#### Violation Breakdown")
            v_counts = violations_only["VIOLATION_TYPE"].value_counts()
            for vtype, count in v_counts.items():
                st.metric(f"{vtype} violations", count)

        # Raw data table
        with st.expander("Raw Detection Data"):
            st.dataframe(detections_df, use_container_width=True, hide_index=True)

            csv_det = detections_df.to_csv(index=False)
            st.download_button(
                "Download Detections CSV",
                data=csv_det,
                file_name="detection_results.csv",
                mime="text/csv",
            )
    else:
        st.info("No detection results found for the selected filters.")

except Exception as e:
    st.warning(f"Could not fetch detection results: {e}")

st.divider()

# --- Violation Timeline ---
st.markdown("### Violation Timeline")

try:
    from inference.result_logger import ResultLogger
    result_logger = ResultLogger()
    violation_df = result_logger.get_violation_stats(days=days_filter)

    if not violation_df.empty:
        fig_timeline = violation_timeline(violation_df)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No violation data available for the selected time range.")
except Exception as e:
    st.warning(f"Could not load violation timeline: {e}")

st.divider()

# --- Drill-down into specific inference ---
st.markdown("### Inference Drill-down")

inference_id_input = st.text_input(
    "Enter Inference ID to view details",
    placeholder="e.g., inf_abc123def456",
)

if inference_id_input:
    try:
        from inference.result_logger import ResultLogger
        result_logger = ResultLogger()
        details_df = result_logger.get_detection_details(inference_id_input)

        if not details_df.empty:
            st.dataframe(details_df, use_container_width=True, hide_index=True)
        else:
            st.info("No detections found for this inference ID.")
    except Exception as e:
        st.error(f"Could not fetch details: {e}")
