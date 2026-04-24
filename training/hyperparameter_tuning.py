"""
Hyperparameter tuning using Optuna with Snowflake logging.
Refactored from the original src/training.py Optuna logic.
"""

import sys
import logging
from pathlib import Path

import torch
import optuna
from optuna.pruners import MedianPruner
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_optuna_tuning(
    data_yaml=None,
    base_model=None,
    epochs_tune=None,
    img_size=None,
    batch_size=None,
    n_trials=None,
    device=None,
):
    """
    Run Optuna hyperparameter tuning for YOLOv8.

    Returns:
        optuna.Study object with best trial results.
    """
    data_yaml = str(data_yaml or Settings.DATA_YAML)
    base_model = base_model or Settings.BASE_MODEL
    epochs_tune = epochs_tune or Settings.TUNING_EPOCHS
    img_size = img_size or Settings.IMAGE_SIZE
    batch_size = batch_size or Settings.BATCH_SIZE
    n_trials = n_trials or Settings.TUNING_TRIALS

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    logger.info("Starting Optuna tuning: %d trials, %d epochs each", n_trials, epochs_tune)
    logger.info("Device: %s", device)

    def objective(trial):
        params = {
            "lr0": trial.suggest_float("lr0", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True),
            "momentum": trial.suggest_float("momentum", 0.85, 0.98),
            "box": trial.suggest_float("box", 6.0, 9.0),
            "cls": trial.suggest_float("cls", 0.5, 2.0),
            "hsv_h": trial.suggest_float("hsv_h", 0.01, 0.04),
            "hsv_s": trial.suggest_float("hsv_s", 0.5, 0.9),
            "scale": trial.suggest_float("scale", 0.3, 0.7),
            "mixup": trial.suggest_float("mixup", 0.0, 0.2),
        }

        model = YOLO(base_model)

        try:
            model.train(
                data=data_yaml,
                epochs=epochs_tune,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                optimizer="AdamW",
                cos_lr=True,
                patience=10,
                amp=True,
                verbose=False,
                plots=False,
                **params,
            )

            metrics = model.val()
            map5095 = metrics.box.map

            trial.report(map5095, step=0)
            return map5095

        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_warmup_steps=5),
    )

    study.optimize(objective, n_trials=n_trials)

    logger.info("Best Trial:")
    logger.info("  mAP50-95: %.4f", study.best_trial.value)
    logger.info("  Best Params: %s", study.best_trial.params)

    return study


if __name__ == "__main__":
    study = run_optuna_tuning()
    print("Best params:", study.best_trial.params)
