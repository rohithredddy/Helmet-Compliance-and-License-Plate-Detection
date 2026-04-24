"""
Data drift monitor.
Compares new data statistics against the training data baseline
to detect significant distribution shifts.
"""

import logging
import pandas as pd

from config.settings import Settings
from config.snowflake_config import SnowflakeManager
from data_preparation.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """Monitor for data distribution drift between training and new data."""

    def __init__(self):
        self.sf = SnowflakeManager()

    def get_baseline_distribution(self):
        """Fetch the baseline class distribution from EDA_STATISTICS."""
        df = self.sf.fetch_dataframe(
            """
            SELECT METRIC_NAME, METRIC_VALUE
            FROM PROCESSED.EDA_STATISTICS
            WHERE CATEGORY = 'class_distribution'
              AND SPLIT = 'all'
              AND METRIC_NAME LIKE 'total_count_%'
            """,
            schema="PROCESSED",
        )
        if df.empty:
            return {}

        result = {}
        for _, row in df.iterrows():
            class_name = row["METRIC_NAME"].replace("total_count_", "")
            result[class_name] = float(row["METRIC_VALUE"])
        return result

    def analyze_new_data(self, new_data_dir):
        """
        Analyze the class distribution of new data and compare with baseline.

        Returns:
            dict with drift metrics and recommendation
        """
        loader = DatasetLoader(data_dir=new_data_dir)

        # Try to load annotations from the new data
        new_annotations = []
        labels_dir = loader.labels_dir
        if not labels_dir.exists():
            # Check if labels are directly in new_data_dir
            from pathlib import Path
            labels_dir = Path(new_data_dir) / "labels"

        if labels_dir.exists():
            for label_file in labels_dir.rglob("*.txt"):
                annotations = loader.parse_label_file(label_file)
                new_annotations.extend(annotations)

        if not new_annotations:
            logger.warning("No annotations found in new data directory.")
            return {"drift_detected": False, "reason": "no_annotations_found"}

        new_df = pd.DataFrame(new_annotations)
        new_counts = new_df["class_name"].value_counts().to_dict()

        baseline = self.get_baseline_distribution()
        if not baseline:
            logger.warning("No baseline distribution found in Snowflake.")
            return {"drift_detected": False, "reason": "no_baseline"}

        # Compute distribution proportions
        baseline_total = sum(baseline.values())
        new_total = sum(new_counts.values())

        drift_scores = {}
        for class_name in Settings.CLASS_NAMES:
            baseline_prop = baseline.get(class_name, 0) / baseline_total if baseline_total > 0 else 0
            new_prop = new_counts.get(class_name, 0) / new_total if new_total > 0 else 0

            drift = abs(baseline_prop - new_prop)
            drift_scores[class_name] = {
                "baseline_proportion": round(baseline_prop, 4),
                "new_proportion": round(new_prop, 4),
                "absolute_drift": round(drift, 4),
            }

        max_drift = max(d["absolute_drift"] for d in drift_scores.values())
        drift_detected = max_drift > 0.15  # Flag if any class shifts by more than 15%

        result = {
            "drift_detected": drift_detected,
            "max_drift": round(max_drift, 4),
            "per_class_drift": drift_scores,
            "new_sample_count": new_total,
            "baseline_total": baseline_total,
            "recommendation": (
                "Significant drift detected. Consider re-tuning hyperparameters."
                if drift_detected
                else "Distribution is stable. Standard retraining is sufficient."
            ),
        }

        logger.info("Drift analysis: max_drift=%.4f, detected=%s", max_drift, drift_detected)
        return result
