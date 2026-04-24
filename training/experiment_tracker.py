"""
Experiment tracker for logging training runs to Snowflake.
Tracks run metadata, per-epoch metrics, and final results.
"""

import uuid
import json
import logging
from datetime import datetime

from config.snowflake_config import SnowflakeManager

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Log training experiments to Snowflake MODELS schema."""

    def __init__(self):
        self.sf = SnowflakeManager()
        self.current_run_id = None

    def start_run(self, model_base, hyperparameters, epochs_planned):
        """Register a new training run. Returns run_id."""
        self.current_run_id = f"run_{uuid.uuid4().hex[:12]}"

        hp_json = json.dumps(hyperparameters).replace("'", "\\'")
        query = (
            f"INSERT INTO TRAINING_RUNS (RUN_ID, MODEL_BASE, HYPERPARAMETERS, EPOCHS_PLANNED, STATUS) "
            f"SELECT '{self.current_run_id}', '{model_base}', PARSE_JSON('{hp_json}'), "
            f"{epochs_planned}, 'RUNNING'"
        )
        self.sf.execute_query(query, schema="MODELS")

        logger.info("Started training run: %s", self.current_run_id)
        return self.current_run_id

    def log_epoch(self, run_id, epoch, metrics):
        """
        Log metrics for a single epoch.
        metrics should be a dict with keys matching TRAINING_METRICS columns.
        """
        row = {
            "RUN_ID": run_id,
            "EPOCH": epoch,
            "TRAIN_BOX_LOSS": metrics.get("train/box_loss"),
            "TRAIN_CLS_LOSS": metrics.get("train/cls_loss"),
            "TRAIN_DFL_LOSS": metrics.get("train/dfl_loss"),
            "VAL_BOX_LOSS": metrics.get("val/box_loss"),
            "VAL_CLS_LOSS": metrics.get("val/cls_loss"),
            "VAL_DFL_LOSS": metrics.get("val/dfl_loss"),
            "PRECISION_B": metrics.get("metrics/precision(B)"),
            "RECALL_B": metrics.get("metrics/recall(B)"),
            "MAP50": metrics.get("metrics/mAP50(B)"),
            "MAP50_95": metrics.get("metrics/mAP50-95(B)"),
            "LEARNING_RATE": metrics.get("lr/pg0"),
            "EPOCH_TIME_SEC": metrics.get("time"),
        }

        self.sf.insert_row(
            table="TRAINING_METRICS",
            data_dict=row,
            schema="MODELS"
        )

    def end_run(self, run_id, best_metrics, epochs_completed, training_time_sec, status="COMPLETED"):
        """Finalize a training run with best metrics."""
        query = """
            UPDATE MODELS.TRAINING_RUNS
            SET EPOCHS_COMPLETED = %s,
                BEST_MAP50 = %s,
                BEST_MAP50_95 = %s,
                BEST_PRECISION = %s,
                BEST_RECALL = %s,
                TRAINING_TIME_SEC = %s,
                STATUS = %s,
                COMPLETED_AT = CURRENT_TIMESTAMP()
            WHERE RUN_ID = %s
        """

        params = (
            epochs_completed,
            best_metrics.get("mAP50"),
            best_metrics.get("mAP50-95"),
            best_metrics.get("precision"),
            best_metrics.get("recall"),
            training_time_sec,
            status,
            run_id,
        )

        self.sf.execute_query(query, schema="MODELS", params=params)
        logger.info("Completed training run: %s (status=%s)", run_id, status)

    def log_existing_results(self, results_csv_path, run_id=None):
        """
        Load an existing results.csv from a previous training run
        and insert all epochs into Snowflake.
        """
        import pandas as pd

        df = pd.read_csv(results_csv_path)
        df.columns = [c.strip() for c in df.columns]

        if run_id is None:
            run_id = self.start_run(
                model_base="yolov8m.pt",
                hyperparameters={"source": "historical", "csv": str(results_csv_path)},
                epochs_planned=len(df)
            )

        for _, row in df.iterrows():
            self.log_epoch(run_id, int(row.get("epoch", 0)), row.to_dict())

        # End run with best metrics from the CSV
        best_idx = df["metrics/mAP50-95(B)"].idxmax() if "metrics/mAP50-95(B)" in df.columns else len(df) - 1
        best_row = df.iloc[best_idx]

        best_metrics = {
            "mAP50": best_row.get("metrics/mAP50(B)"),
            "mAP50-95": best_row.get("metrics/mAP50-95(B)"),
            "precision": best_row.get("metrics/precision(B)"),
            "recall": best_row.get("metrics/recall(B)"),
        }

        total_time = df["time"].iloc[-1] if "time" in df.columns else 0
        self.end_run(run_id, best_metrics, len(df), total_time)

        logger.info("Logged %d epochs from existing results CSV.", len(df))
        return run_id
