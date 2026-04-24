"""
EDA Dashboard page.
Displays class distributions, bbox statistics, and image quality analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from components.snowflake_auth import require_snowflake
from components.charts import (
    class_distribution_bar,
    class_distribution_pie,
)
from config.settings import Settings

st.set_page_config(page_title="EDA Dashboard", layout="wide")
st.markdown("## EDA Dashboard")
st.markdown("Exploratory data analysis of the helmet detection dataset.")

sf = require_snowflake()

# Option to regenerate EDA stats
with st.expander("Regenerate EDA Statistics"):
    st.markdown("Run fresh EDA analysis and upload results to Snowflake.")
    if st.button("Run EDA Pipeline"):
        with st.spinner("Running EDA analysis (this may take a few minutes)..."):
            try:
                from eda.generate_eda_report import generate_eda_report
                report = generate_eda_report(upload_to_snowflake=True)
                st.success(f"EDA complete. Generated {report['total_records']} statistics.")
            except Exception as e:
                st.error(f"EDA failed: {e}")

st.divider()

# Fetch EDA statistics from Snowflake
try:
    eda_df = sf.fetch_dataframe(
        "SELECT * FROM PROCESSED.EDA_STATISTICS ORDER BY CATEGORY, METRIC_NAME",
        schema="PROCESSED",
    )
except Exception as e:
    st.warning(f"Could not fetch EDA stats from Snowflake: {e}")
    st.info("Click 'Run EDA Pipeline' above to generate statistics.")
    st.stop()

if eda_df.empty:
    st.info("No EDA statistics found. Click 'Run EDA Pipeline' above to generate them.")
    st.stop()

# --- Class Distribution ---
st.markdown("### Class Distribution")

class_df = eda_df[eda_df["CATEGORY"] == "class_distribution"]

col1, col2 = st.columns(2)

with col1:
    # Per-split counts
    split_counts = class_df[
        class_df["METRIC_NAME"].str.startswith("count_") &
        (class_df["SPLIT"] != "all")
    ].copy()
    if not split_counts.empty:
        split_counts["class_name"] = split_counts["METRIC_NAME"].str.replace("count_", "")
        split_counts = split_counts.rename(columns={"METRIC_VALUE": "count", "SPLIT": "split"})
        fig = class_distribution_bar(split_counts[["class_name", "split", "count"]])
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Total counts pie
    total_rows = class_df[class_df["METRIC_NAME"].str.startswith("total_count_")]
    if not total_rows.empty:
        totals = {}
        for _, row in total_rows.iterrows():
            name = row["METRIC_NAME"].replace("total_count_", "")
            totals[name] = row["METRIC_VALUE"]
        fig = class_distribution_pie(totals)
        st.plotly_chart(fig, use_container_width=True)

# Key metrics
metric_cols = st.columns(4)
imbalance = class_df[class_df["METRIC_NAME"] == "imbalance_ratio"]
if not imbalance.empty:
    metric_cols[0].metric("Imbalance Ratio", f"{imbalance.iloc[0]['METRIC_VALUE']:.2f}")

obj_stats = class_df[class_df["METRIC_NAME"].str.startswith("objects_per_image_")]
for idx, (_, row) in enumerate(obj_stats.iterrows()):
    if idx < 3:
        label = row["METRIC_NAME"].replace("objects_per_image_", "Obj/Img ")
        metric_cols[idx + 1].metric(label.title(), f"{row['METRIC_VALUE']:.1f}")

st.divider()

# --- Bounding Box Analysis ---
st.markdown("### Bounding Box Statistics")

bbox_df = eda_df[eda_df["CATEGORY"] == "bbox_statistics"]
if not bbox_df.empty:
    # Parse into a readable table
    bbox_data = []
    for class_name in Settings.CLASS_NAMES:
        row_data = {"Class": class_name}
        for _, row in bbox_df.iterrows():
            if row["METRIC_NAME"].startswith(class_name):
                metric = row["METRIC_NAME"].replace(f"{class_name}_", "")
                row_data[metric] = round(row["METRIC_VALUE"], 4)
        if len(row_data) > 1:
            bbox_data.append(row_data)

    if bbox_data:
        bbox_table = pd.DataFrame(bbox_data)
        st.dataframe(bbox_table, use_container_width=True, hide_index=True)

# Size distribution
size_df = eda_df[eda_df["CATEGORY"] == "bbox_size_distribution"]
if not size_df.empty:
    st.markdown("#### Object Size Distribution")
    size_data = []
    for _, row in size_df.iterrows():
        parts = row["METRIC_NAME"].rsplit("_", 1)
        if len(parts) == 2:
            size_data.append({
                "Class": parts[0],
                "Size": parts[1],
                "Count": int(row["METRIC_VALUE"]),
            })
    if size_data:
        size_table = pd.DataFrame(size_data)
        import plotly.express as px
        fig = px.bar(
            size_table, x="Class", y="Count", color="Size",
            barmode="group", title="Object Size Categories",
            color_discrete_sequence=["#636EFA", "#00CC96", "#EF553B"],
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Image Quality ---
st.markdown("### Image Quality Metrics")

quality_df = eda_df[eda_df["CATEGORY"] == "image_quality"]
if not quality_df.empty:
    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.markdown("#### Resolution Statistics")
        res_data = quality_df[quality_df["METRIC_NAME"].str.startswith("resolution_")]
        if not res_data.empty:
            res_table = []
            for _, row in res_data.iterrows():
                metric = row["METRIC_NAME"].replace("resolution_", "")
                res_table.append({
                    "Split": row["SPLIT"],
                    "Metric": metric,
                    "Value": round(row["METRIC_VALUE"], 1),
                })
            st.dataframe(pd.DataFrame(res_table), use_container_width=True, hide_index=True)

    with col_q2:
        st.markdown("#### Brightness & Lighting")
        brightness_data = quality_df[
            quality_df["METRIC_NAME"].isin(["mean_brightness", "night_image_percentage"])
        ]
        if not brightness_data.empty:
            for _, row in brightness_data.iterrows():
                label = row["METRIC_NAME"].replace("_", " ").title()
                suffix = "%" if "percentage" in row["METRIC_NAME"] else ""
                st.metric(
                    f"{label} ({row['SPLIT']})",
                    f"{row['METRIC_VALUE']:.1f}{suffix}",
                )
