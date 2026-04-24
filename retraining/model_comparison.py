"""
Model comparison utility.
Compares two models on the same test set and determines which is better.
"""

import logging
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from config.settings import Settings

logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare two YOLO models on the test dataset."""

    def __init__(self, data_yaml=None):
        self.data_yaml = str(data_yaml or Settings.DATA_YAML)

    def evaluate_model(self, model_path, split="test"):
        """
        Evaluate a model on a dataset split.

        Returns:
            dict of evaluation metrics
        """
        logger.info("Evaluating model: %s on '%s' split", model_path, split)
        model = YOLO(model_path)
        metrics = model.val(data=self.data_yaml, split=split)

        results = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

        # Per-class AP if available
        per_class = {}
        if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
            for i, class_name in enumerate(Settings.CLASS_NAMES):
                if i < len(metrics.box.ap50):
                    per_class[class_name] = float(metrics.box.ap50[i])

        results["per_class_ap50"] = per_class

        logger.info("Results: mAP50=%.4f, mAP50-95=%.4f", results["mAP50"], results["mAP50-95"])
        return results

    def compare(self, model_a_path, model_b_path,
                model_a_name="model_a", model_b_name="model_b"):
        """
        Compare two models and determine which is better.

        Returns:
            dict with comparison results and winner
        """
        logger.info("Comparing %s vs %s", model_a_name, model_b_name)

        metrics_a = self.evaluate_model(model_a_path)
        metrics_b = self.evaluate_model(model_b_path)

        # Build comparison table
        comparison = {
            "model_a": {"name": model_a_name, "path": model_a_path, **metrics_a},
            "model_b": {"name": model_b_name, "path": model_b_path, **metrics_b},
        }

        # Determine winner based on mAP50-95 (primary) and mAP50 (secondary)
        score_a = metrics_a["mAP50-95"] * 0.7 + metrics_a["mAP50"] * 0.3
        score_b = metrics_b["mAP50-95"] * 0.7 + metrics_b["mAP50"] * 0.3

        if score_b > score_a:
            winner = model_b_name
            margin = score_b - score_a
        else:
            winner = model_a_name
            margin = score_a - score_b

        comparison["winner"] = winner
        comparison["margin"] = round(margin, 4)
        comparison["score_a"] = round(score_a, 4)
        comparison["score_b"] = round(score_b, 4)

        logger.info(
            "Winner: %s (margin: %.4f, score_a: %.4f, score_b: %.4f)",
            winner, margin, score_a, score_b,
        )

        return comparison

    def get_comparison_dataframe(self, comparison):
        """Convert comparison dict to a display-friendly DataFrame."""
        rows = []
        for key in ["mAP50", "mAP50-95", "precision", "recall"]:
            rows.append({
                "Metric": key,
                comparison["model_a"]["name"]: comparison["model_a"].get(key),
                comparison["model_b"]["name"]: comparison["model_b"].get(key),
                "Difference": (
                    comparison["model_b"].get(key, 0) - comparison["model_a"].get(key, 0)
                ),
            })
        return pd.DataFrame(rows)
