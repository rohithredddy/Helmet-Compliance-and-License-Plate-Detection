"""
Image quality analysis for the dataset.
Analyzes resolution, brightness, contrast, and file sizes.
"""

import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import Settings
from data_preparation.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class ImageQualityAnalyzer:
    """Analyze image quality metrics across the dataset."""

    def __init__(self, loader=None):
        self.loader = loader or DatasetLoader()
        self.data_dir = self.loader.data_dir

    def compute_resolution_stats(self):
        """Compute image resolution statistics per split."""
        dataset_df = self.loader.load_all_splits()
        if dataset_df.empty:
            return {}

        stats = {}
        for split in dataset_df["split"].unique():
            split_df = dataset_df[dataset_df["split"] == split]
            stats[split] = {
                "mean_width": float(split_df["image_width"].mean()),
                "mean_height": float(split_df["image_height"].mean()),
                "min_width": int(split_df["image_width"].min()),
                "max_width": int(split_df["image_width"].max()),
                "min_height": int(split_df["image_height"].min()),
                "max_height": int(split_df["image_height"].max()),
            }
        return stats

    def compute_brightness_stats(self, sample_size=100):
        """
        Compute average brightness for a sample of images per split.
        Brightness = mean pixel value in grayscale.
        """
        results = []

        for split in Settings.SPLITS:
            img_dir = self.data_dir / "images" / split
            if not img_dir.exists():
                continue

            image_files = [
                f for f in img_dir.iterdir()
                if f.suffix.lower() in Settings.IMAGE_EXTENSIONS
            ]

            sample = image_files[:sample_size]

            for img_path in sample:
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        brightness = float(np.mean(img))
                        results.append({
                            "image_filename": img_path.name,
                            "split": split,
                            "brightness": brightness,
                            "is_night": brightness < 80,
                        })
                except Exception as e:
                    logger.warning("Could not process %s: %s", img_path.name, e)

        return pd.DataFrame(results)

    def compute_file_size_stats(self):
        """Compute file size statistics per split."""
        dataset_df = self.loader.load_all_splits()
        if dataset_df.empty:
            return {}

        stats = {}
        for split in dataset_df["split"].unique():
            split_df = dataset_df[dataset_df["split"] == split]
            sizes = split_df["file_size_bytes"]
            stats[split] = {
                "mean_size_kb": float(sizes.mean() / 1024),
                "median_size_kb": float(sizes.median() / 1024),
                "min_size_kb": float(sizes.min() / 1024),
                "max_size_kb": float(sizes.max() / 1024),
                "total_size_mb": float(sizes.sum() / (1024 * 1024)),
            }
        return stats

    def get_eda_records(self):
        """Generate EDA records for Snowflake insertion."""
        records = []

        # Resolution stats
        res_stats = self.compute_resolution_stats()
        for split, stats in res_stats.items():
            for metric_name, value in stats.items():
                records.append({
                    "category": "image_quality",
                    "metric_name": f"resolution_{metric_name}",
                    "metric_value": float(value),
                    "split": split,
                })

        # Brightness stats
        brightness_df = self.compute_brightness_stats()
        if not brightness_df.empty:
            for split in brightness_df["split"].unique():
                split_df = brightness_df[brightness_df["split"] == split]
                records.append({
                    "category": "image_quality",
                    "metric_name": "mean_brightness",
                    "metric_value": float(split_df["brightness"].mean()),
                    "split": split,
                })
                night_pct = float(split_df["is_night"].mean() * 100)
                records.append({
                    "category": "image_quality",
                    "metric_name": "night_image_percentage",
                    "metric_value": night_pct,
                    "split": split,
                })

        # File size stats
        size_stats = self.compute_file_size_stats()
        for split, stats in size_stats.items():
            for metric_name, value in stats.items():
                records.append({
                    "category": "image_quality",
                    "metric_name": f"filesize_{metric_name}",
                    "metric_value": value,
                    "split": split,
                })

        return records
