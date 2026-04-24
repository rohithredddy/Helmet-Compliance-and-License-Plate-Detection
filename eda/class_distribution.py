"""
Class distribution analysis for the YOLO dataset.
Computes per-class counts, imbalance ratios, and split-level breakdowns.
"""

import logging
import pandas as pd

from config.settings import Settings
from data_preparation.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class ClassDistributionAnalyzer:
    """Analyze class distributions across dataset splits."""

    def __init__(self, loader=None):
        self.loader = loader or DatasetLoader()

    def compute_class_counts(self):
        """
        Compute total class instance counts across all splits.
        Returns a DataFrame with columns: class_name, split, count
        """
        annotations_df = self.loader.load_all_annotations()
        if annotations_df.empty:
            return pd.DataFrame(columns=["class_name", "split", "count"])

        counts = (
            annotations_df
            .groupby(["split", "class_name"])
            .size()
            .reset_index(name="count")
        )
        return counts

    def compute_total_counts(self):
        """Compute total counts per class (all splits combined)."""
        annotations_df = self.loader.load_all_annotations()
        if annotations_df.empty:
            return {}

        return annotations_df["class_name"].value_counts().to_dict()

    def compute_imbalance_ratio(self):
        """
        Compute the imbalance ratio (max_class / min_class).
        A ratio close to 1.0 means balanced, higher means imbalanced.
        """
        totals = self.compute_total_counts()
        if not totals:
            return None

        counts = list(totals.values())
        max_count = max(counts)
        min_count = min(counts) if min(counts) > 0 else 1
        ratio = max_count / min_count

        logger.info("Class imbalance ratio: %.2f (max=%d, min=%d)", ratio, max_count, min_count)
        return ratio

    def compute_objects_per_image(self):
        """Compute distribution of object counts per image."""
        dataset_df = self.loader.load_all_splits()
        if dataset_df.empty:
            return {}

        stats = {
            "mean": float(dataset_df["num_objects"].mean()),
            "median": float(dataset_df["num_objects"].median()),
            "min": int(dataset_df["num_objects"].min()),
            "max": int(dataset_df["num_objects"].max()),
            "std": float(dataset_df["num_objects"].std()),
        }
        return stats

    def get_eda_records(self):
        """
        Generate EDA statistic records suitable for Snowflake insertion.
        Returns list of dicts with: category, metric_name, metric_value, split
        """
        records = []

        # Per-split class counts
        counts_df = self.compute_class_counts()
        for _, row in counts_df.iterrows():
            records.append({
                "category": "class_distribution",
                "metric_name": f"count_{row['class_name']}",
                "metric_value": float(row["count"]),
                "split": row["split"],
            })

        # Total counts
        totals = self.compute_total_counts()
        for class_name, count in totals.items():
            records.append({
                "category": "class_distribution",
                "metric_name": f"total_count_{class_name}",
                "metric_value": float(count),
                "split": "all",
            })

        # Imbalance ratio
        ratio = self.compute_imbalance_ratio()
        if ratio is not None:
            records.append({
                "category": "class_distribution",
                "metric_name": "imbalance_ratio",
                "metric_value": ratio,
                "split": "all",
            })

        # Objects per image stats
        obj_stats = self.compute_objects_per_image()
        for stat_name, value in obj_stats.items():
            records.append({
                "category": "class_distribution",
                "metric_name": f"objects_per_image_{stat_name}",
                "metric_value": value,
                "split": "all",
            })

        return records
