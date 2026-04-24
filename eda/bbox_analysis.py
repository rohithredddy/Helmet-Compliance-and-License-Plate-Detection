"""
Bounding box analysis for YOLO-format dataset.
Analyzes bbox dimensions, aspect ratios, and spatial distributions.
"""

import logging
import numpy as np
import pandas as pd

from config.settings import Settings
from data_preparation.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class BBoxAnalyzer:
    """Analyze bounding box characteristics across the dataset."""

    def __init__(self, loader=None):
        self.loader = loader or DatasetLoader()

    def compute_bbox_statistics(self):
        """
        Compute width, height, area, and aspect ratio statistics per class.
        All values are in normalized coordinates (0-1).
        """
        ann_df = self.loader.load_all_annotations()
        if ann_df.empty:
            return pd.DataFrame()

        ann_df["area"] = ann_df["bbox_width"] * ann_df["bbox_height"]
        ann_df["aspect_ratio"] = ann_df["bbox_width"] / ann_df["bbox_height"].replace(0, np.nan)

        stats = (
            ann_df
            .groupby("class_name")
            .agg({
                "bbox_width": ["mean", "std", "min", "max"],
                "bbox_height": ["mean", "std", "min", "max"],
                "area": ["mean", "std", "min", "max"],
                "aspect_ratio": ["mean", "std"],
            })
        )

        # Flatten MultiIndex columns
        stats.columns = ["_".join(col) for col in stats.columns]
        stats = stats.reset_index()

        logger.info("Computed bbox statistics for %d classes", len(stats))
        return stats

    def compute_size_distribution(self):
        """
        Categorize bounding boxes into small/medium/large based on area.
        Uses COCO-style thresholds adapted to normalized coords.
        """
        ann_df = self.loader.load_all_annotations()
        if ann_df.empty:
            return {}

        ann_df["area"] = ann_df["bbox_width"] * ann_df["bbox_height"]

        def categorize(area):
            if area < 0.01:
                return "small"
            elif area < 0.05:
                return "medium"
            else:
                return "large"

        ann_df["size_category"] = ann_df["area"].apply(categorize)

        distribution = (
            ann_df
            .groupby(["class_name", "size_category"])
            .size()
            .reset_index(name="count")
        )
        return distribution

    def compute_center_heatmap_data(self):
        """
        Get bounding box center coordinates for heatmap visualization.
        Returns DataFrame with x_center, y_center, class_name.
        """
        ann_df = self.loader.load_all_annotations()
        if ann_df.empty:
            return pd.DataFrame()

        return ann_df[["x_center", "y_center", "class_name"]].copy()

    def get_eda_records(self):
        """Generate EDA records for Snowflake insertion."""
        records = []

        bbox_stats = self.compute_bbox_statistics()
        for _, row in bbox_stats.iterrows():
            class_name = row["class_name"]
            for col in bbox_stats.columns:
                if col == "class_name":
                    continue
                records.append({
                    "category": "bbox_statistics",
                    "metric_name": f"{class_name}_{col}",
                    "metric_value": float(row[col]) if pd.notna(row[col]) else 0.0,
                    "split": "all",
                })

        size_dist = self.compute_size_distribution()
        if isinstance(size_dist, pd.DataFrame) and not size_dist.empty:
            for _, row in size_dist.iterrows():
                records.append({
                    "category": "bbox_size_distribution",
                    "metric_name": f"{row['class_name']}_{row['size_category']}",
                    "metric_value": float(row["count"]),
                    "split": "all",
                })

        return records
