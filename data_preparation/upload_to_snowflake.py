"""
Upload dataset metadata and configs to Snowflake.
Populates the RAW_DATA.DATASET_METADATA table with image-level stats.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from config.snowflake_config import SnowflakeManager
from data_preparation.dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def upload_dataset_metadata():
    """Load dataset info and insert into Snowflake DATASET_METADATA table."""
    loader = DatasetLoader()
    df = loader.load_all_splits()

    if df.empty:
        logger.error("No dataset records found. Aborting upload.")
        return

    logger.info("Uploading %d records to Snowflake...", len(df))

    sf = SnowflakeManager()

    # Clear existing records to avoid duplicates
    sf.execute_query(
        "DELETE FROM RAW_DATA.DATASET_METADATA",
        schema="RAW_DATA"
    )

    # Rename columns to match table schema (uppercase)
    df.columns = [c.upper() for c in df.columns]

    sf.insert_dataframe(
        table="DATASET_METADATA",
        df=df,
        schema="RAW_DATA"
    )

    logger.info("Dataset metadata uploaded successfully.")

    # Also upload the data.yaml to the raw images stage
    yaml_path = str(Settings.DATA_YAML).replace("\\", "/")
    try:
        sf.upload_file_to_stage(
            local_path=yaml_path,
            stage_name="RAW_IMAGES_STAGE",
            schema="RAW_DATA"
        )
        logger.info("data.yaml uploaded to @RAW_IMAGES_STAGE")
    except Exception as e:
        logger.warning("Could not upload data.yaml to stage: %s", e)


if __name__ == "__main__":
    upload_dataset_metadata()
