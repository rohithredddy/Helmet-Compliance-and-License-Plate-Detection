"""
Reusable chart components for the Streamlit dashboard.
Built with Plotly for interactive visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def class_distribution_bar(counts_df, title="Class Distribution"):
    """Bar chart of class counts grouped by split."""
    fig = px.bar(
        counts_df,
        x="class_name",
        y="count",
        color="split",
        barmode="group",
        title=title,
        color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"],
        labels={"class_name": "Class", "count": "Count", "split": "Split"},
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def class_distribution_pie(totals_dict, title="Overall Class Distribution"):
    """Pie chart of total class counts."""
    fig = px.pie(
        names=list(totals_dict.keys()),
        values=list(totals_dict.values()),
        title=title,
        color_discrete_sequence=["#00CC96", "#636EFA", "#EF553B", "#AB63FA"],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def training_loss_curves(metrics_df):
    """Multi-line chart showing training loss over epochs."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Box Loss", "Cls Loss", "DFL Loss"))

    loss_cols = [
        ("TRAIN_BOX_LOSS", "VAL_BOX_LOSS"),
        ("TRAIN_CLS_LOSS", "VAL_CLS_LOSS"),
        ("TRAIN_DFL_LOSS", "VAL_DFL_LOSS"),
    ]

    colors = [("#636EFA", "#EF553B"), ("#636EFA", "#EF553B"), ("#636EFA", "#EF553B")]

    for i, (train_col, val_col) in enumerate(loss_cols, 1):
        if train_col in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df["EPOCH"], y=metrics_df[train_col],
                    name=f"Train", mode="lines",
                    line=dict(color=colors[i-1][0]),
                    showlegend=(i == 1),
                ),
                row=1, col=i,
            )
        if val_col in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df["EPOCH"], y=metrics_df[val_col],
                    name=f"Val", mode="lines",
                    line=dict(color=colors[i-1][1], dash="dash"),
                    showlegend=(i == 1),
                ),
                row=1, col=i,
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        title_text="Training Loss Curves",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def validation_metrics_chart(metrics_df):
    """Line chart of mAP50, mAP50-95, precision, recall over epochs."""
    fig = go.Figure()

    metric_configs = [
        ("MAP50", "mAP50", "#00CC96"),
        ("MAP50_95", "mAP50-95", "#636EFA"),
        ("PRECISION_B", "Precision", "#EF553B"),
        ("RECALL_B", "Recall", "#AB63FA"),
    ]

    for col, name, color in metric_configs:
        if col in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df["EPOCH"], y=metrics_df[col],
                    name=name, mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                )
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Validation Metrics Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Score",
        height=450,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def violation_timeline(violation_df):
    """Bar chart of violations over time."""
    if violation_df.empty:
        return go.Figure().update_layout(
            title="No violation data available",
            template="plotly_dark",
        )

    fig = px.bar(
        violation_df,
        x="DATE",
        y="COUNT",
        color="VIOLATION_TYPE",
        title="Violations Over Time",
        barmode="stack",
        color_discrete_sequence=["#EF553B", "#FFA15A"],
        labels={"DATE": "Date", "COUNT": "Count", "VIOLATION_TYPE": "Type"},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def detection_confidence_histogram(detections_df):
    """Histogram of detection confidences by class."""
    if detections_df.empty:
        return go.Figure()

    fig = px.histogram(
        detections_df,
        x="CONFIDENCE",
        color="CLASS_NAME",
        nbins=30,
        title="Detection Confidence Distribution",
        color_discrete_sequence=["#00CC96", "#636EFA", "#EF553B", "#AB63FA"],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig
