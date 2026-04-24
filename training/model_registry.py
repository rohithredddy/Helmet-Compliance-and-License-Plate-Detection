"""
Model registry for managing trained model versions in Snowflake.
Uploads weights to stage and tracks versions in MODEL_REGISTRY table.
"""

import os
import logging

from config.settings import Settings
from config.snowflake_config import SnowflakeManager

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manage model versions in Snowflake."""

    def __init__(self):
        self.sf = SnowflakeManager()

    def register_model(self, weights_path, model_version, run_id=None,
                       model_base=None, metrics=None, notes=None):
        """
        Register a trained model:
        1. Upload weights to MODEL_WEIGHTS_STAGE
        2. Insert record into MODEL_REGISTRY table
        """
        weights_path = str(weights_path).replace("\\", "/")
        file_size = os.path.getsize(weights_path) if os.path.exists(weights_path) else 0
        stage_path = f"@MODELS.MODEL_WEIGHTS_STAGE/{model_version}/best.pt"

        # Upload weights to stage
        try:
            self.sf.upload_file_to_stage(
                local_path=weights_path,
                stage_name=f"MODEL_WEIGHTS_STAGE/{model_version}",
                schema="MODELS",
                auto_compress=False
            )
            logger.info("Uploaded model weights to %s", stage_path)
        except Exception as e:
            logger.error("Failed to upload weights: %s", e)
            stage_path = f"local://{weights_path}"

        metrics = metrics or {}

        self.sf.insert_row(
            table="MODEL_REGISTRY",
            data_dict={
                "MODEL_VERSION": model_version,
                "RUN_ID": run_id,
                "STAGE_PATH": stage_path,
                "MODEL_BASE": model_base or Settings.BASE_MODEL,
                "MAP50": metrics.get("mAP50"),
                "MAP50_95": metrics.get("mAP50-95"),
                "PRECISION_VAL": metrics.get("precision"),
                "RECALL_VAL": metrics.get("recall"),
                "FILE_SIZE_BYTES": file_size,
                "IS_ACTIVE": False,
                "NOTES": notes,
            },
            schema="MODELS"
        )

        logger.info("Registered model version: %s", model_version)
        return model_version

    def promote_model(self, model_version):
        """Set a model version as the active (deployed) model."""
        # Deactivate all models
        self.sf.execute_query(
            "UPDATE MODELS.MODEL_REGISTRY SET IS_ACTIVE = FALSE",
            schema="MODELS"
        )

        # Activate the target version
        self.sf.execute_query(
            "UPDATE MODELS.MODEL_REGISTRY SET IS_ACTIVE = TRUE WHERE MODEL_VERSION = %s",
            schema="MODELS",
            params=(model_version,)
        )

        logger.info("Promoted model version: %s", model_version)

    def get_active_model(self):
        """Get the currently active model version."""
        columns, rows = self.sf.execute_query(
            "SELECT MODEL_VERSION, STAGE_PATH, MAP50, MAP50_95 "
            "FROM MODELS.MODEL_REGISTRY WHERE IS_ACTIVE = TRUE",
            schema="MODELS"
        )
        if rows:
            return {
                "model_version": rows[0][0],
                "stage_path": rows[0][1],
                "mAP50": rows[0][2],
                "mAP50-95": rows[0][3],
            }
        return None

    def download_active_model(self, download_dir):
        """Download the active model weights from Snowflake to a local directory."""
        active = self.get_active_model()
        if not active:
            logger.error("No active model found in registry.")
            return None

        stage_path = active["stage_path"]
        if stage_path.startswith("local://"):
            local_path = stage_path.replace("local://", "")
            logger.info("Active model is local: %s", local_path)
            return local_path

        # Snowflake stage path
        logger.info("Downloading active model from %s to %s", stage_path, download_dir)
        os.makedirs(download_dir, exist_ok=True)
        self.sf.download_file_from_stage(stage_path, str(download_dir), schema="MODELS")
        
        # Snowflake GET downloads to <download_dir>/best.pt
        expected_path = os.path.join(download_dir, "best.pt")
        if os.path.exists(expected_path):
            return expected_path
        return None

    def list_models(self):
        """List all registered model versions."""
        df = self.sf.fetch_dataframe(
            "SELECT MODEL_VERSION, RUN_ID, MAP50, MAP50_95, PRECISION_VAL, "
            "RECALL_VAL, IS_ACTIVE, REGISTERED_AT, NOTES "
            "FROM MODELS.MODEL_REGISTRY ORDER BY REGISTERED_AT DESC",
            schema="MODELS"
        )
        return df

    def register_existing_model(self):
        """
        Register the existing best.pt model from the results directory.
        This is called once during initial setup.
        """
        import pandas as pd

        weights_path = Settings.DEFAULT_MODEL_PATH
        if not weights_path.exists():
            logger.error("Model weights not found at %s", weights_path)
            return None

        # Read metrics from results.csv
        results_csv = Settings.RESULTS_CSV
        metrics = {}
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            best_idx = df["metrics/mAP50-95(B)"].idxmax()
            best_row = df.iloc[best_idx]
            metrics = {
                "mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(best_row.get("metrics/mAP50-95(B)", 0)),
                "precision": float(best_row.get("metrics/precision(B)", 0)),
                "recall": float(best_row.get("metrics/recall(B)", 0)),
            }

        version = self.register_model(
            weights_path=weights_path,
            model_version="v1.0_initial",
            model_base="yolov8m.pt",
            metrics=metrics,
            notes="Initial model trained with Optuna-tuned hyperparameters. 26 epochs."
        )

        self.promote_model(version)
        return version
