"""
EDA report generator.
Orchestrates all EDA modules and saves results to Snowflake.
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import Settings
from config.snowflake_config import SnowflakeManager
from eda.class_distribution import ClassDistributionAnalyzer
from eda.bbox_analysis import BBoxAnalyzer
from eda.image_quality import ImageQualityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_eda_report(upload_to_snowflake=True):
    """
    Run all EDA analyses and optionally upload results to Snowflake.
    Returns a dict of all computed statistics.
    """
    logger.info("Starting EDA report generation...")

    # Run analyzers
    class_analyzer = ClassDistributionAnalyzer()
    bbox_analyzer = BBoxAnalyzer()
    quality_analyzer = ImageQualityAnalyzer()

    all_records = []

    logger.info("Running class distribution analysis...")
    all_records.extend(class_analyzer.get_eda_records())

    logger.info("Running bounding box analysis...")
    all_records.extend(bbox_analyzer.get_eda_records())

    logger.info("Running image quality analysis...")
    all_records.extend(quality_analyzer.get_eda_records())

    logger.info("Generated %d EDA records total.", len(all_records))

    # Upload to Snowflake
    if upload_to_snowflake and all_records:
        try:
            sf = SnowflakeManager()

            # Clear old EDA stats
            sf.execute_query(
                "DELETE FROM PROCESSED.EDA_STATISTICS",
                schema="PROCESSED"
            )

            df = pd.DataFrame(all_records)
            df.columns = [c.upper() for c in df.columns]

            sf.insert_dataframe(
                table="EDA_STATISTICS",
                df=df,
                schema="PROCESSED"
            )
            logger.info("EDA statistics uploaded to Snowflake successfully.")
        except Exception as e:
            logger.error("Failed to upload EDA stats to Snowflake: %s", e)

    # Build summary
    summary = {
        "total_records": len(all_records),
        "categories": list(set(r["category"] for r in all_records)),
        "class_totals": class_analyzer.compute_total_counts(),
        "imbalance_ratio": class_analyzer.compute_imbalance_ratio(),
        "objects_per_image": class_analyzer.compute_objects_per_image(),
    }

    logger.info("EDA Report Summary:")
    logger.info("  Categories: %s", summary["categories"])
    logger.info("  Class totals: %s", summary["class_totals"])
    logger.info("  Imbalance ratio: %s", summary.get("imbalance_ratio"))

    return summary


if __name__ == "__main__":
    report = generate_eda_report(upload_to_snowflake=True)
    print(json.dumps(report, indent=2, default=str))
