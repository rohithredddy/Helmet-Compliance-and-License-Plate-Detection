"""
Training script for YOLOv8 helmet detection model.
Refactored from the original src/training.py with Snowflake experiment tracking.
"""

import sys
import time
import logging
from pathlib import Path

import torch
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from training.experiment_tracker import ExperimentTracker
from training.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_model(
    data_yaml=None,
    base_model=None,
    epochs=None,
    img_size=None,
    batch_size=None,
    hyperparams=None,
    experiment_name="helmet_detection",
    device=None,
    track_to_snowflake=True,
):
    """
    Train a YOLOv8 model with full experiment tracking.

    Args:
        data_yaml: Path to dataset YAML config
        base_model: Base model to fine-tune from
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Training batch size
        hyperparams: Dict of additional hyperparameters (lr0, momentum, etc.)
        experiment_name: Name for this training run
        device: CUDA device or 'cpu'
        track_to_snowflake: Whether to log to Snowflake

    Returns:
        Trained YOLO model object
    """
    data_yaml = str(data_yaml or Settings.DATA_YAML)
    base_model = base_model or Settings.BASE_MODEL
    epochs = epochs or Settings.DEFAULT_EPOCHS
    img_size = img_size or Settings.IMAGE_SIZE
    batch_size = batch_size or Settings.BATCH_SIZE
    hyperparams = hyperparams or {}

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    logger.info("Device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Initialize experiment tracking
    tracker = None
    run_id = None
    if track_to_snowflake:
        try:
            tracker = ExperimentTracker()
            all_params = {
                "base_model": base_model,
                "epochs": epochs,
                "img_size": img_size,
                "batch_size": batch_size,
                "experiment_name": experiment_name,
                **hyperparams,
            }
            run_id = tracker.start_run(base_model, all_params, epochs)
        except Exception as e:
            logger.warning("Could not initialize Snowflake tracking: %s", e)
            tracker = None

    # Train
    model = YOLO(base_model)
    start_time = time.time()

    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            optimizer="AdamW",
            cos_lr=True,
            patience=Settings.PATIENCE,
            amp=True,
            name=experiment_name,
            plots=True,
            **hyperparams,
        )

        training_time = time.time() - start_time

        # Validate
        metrics = model.val()
        best_metrics = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

        logger.info("Training complete in %.1f seconds", training_time)
        logger.info("Best metrics: %s", best_metrics)

        # Log to Snowflake
        if tracker and run_id:
            try:
                # Log epoch-level metrics from results.csv
                results_csv = Path(model.trainer.save_dir) / "results.csv"
                if results_csv.exists():
                    df = pd.read_csv(results_csv)
                    df.columns = [c.strip() for c in df.columns]
                    for _, row in df.iterrows():
                        tracker.log_epoch(run_id, int(row.get("epoch", 0)), row.to_dict())

                tracker.end_run(run_id, best_metrics, epochs, training_time)

                # Register model
                registry = ModelRegistry()
                best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
                if best_weights.exists():
                    version = f"v_{run_id}"
                    registry.register_model(
                        weights_path=best_weights,
                        model_version=version,
                        run_id=run_id,
                        model_base=base_model,
                        metrics=best_metrics,
                    )
            except Exception as e:
                logger.warning("Error logging to Snowflake: %s", e)

        return model

    except Exception as e:
        training_time = time.time() - start_time
        logger.error("Training failed: %s", e)

        if tracker and run_id:
            try:
                tracker.end_run(run_id, {}, 0, training_time, status="FAILED")
            except Exception:
                pass
        raise


if __name__ == "__main__":
    train_model(track_to_snowflake=True)
