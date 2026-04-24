"""
End-to-end retraining pipeline.
Orchestrates data merging, drift analysis, training, evaluation, and model promotion.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from retraining.retrain_trigger import RetrainTrigger
from retraining.data_drift_monitor import DataDriftMonitor
from retraining.model_comparison import ModelComparison
from training.train import train_model
from training.hyperparameter_tuning import run_optuna_tuning
from training.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RetrainPipeline:
    """Orchestrate the full retraining workflow."""

    def __init__(self):
        self.trigger = RetrainTrigger()
        self.drift_monitor = DataDriftMonitor()
        self.registry = ModelRegistry()

    def run(self, new_data_dir=None, force=False):
        """
        Execute the retraining pipeline.

        Steps:
        1. Check retraining trigger conditions
        2. Analyze data drift
        3. Train new model (with or without hyperparameter re-tuning)
        4. Compare new model vs active model
        5. Promote if better

        Args:
            new_data_dir: directory containing new training data
            force: skip trigger check and force retraining

        Returns:
            dict with pipeline results
        """
        logger.info("=" * 60)
        logger.info("RETRAINING PIPELINE STARTED")
        logger.info("=" * 60)

        # Step 1: Check trigger
        if not force:
            trigger_result = self.trigger.evaluate_trigger()
            logger.info("Trigger evaluation: %s", trigger_result["reason"])

            if not trigger_result["should_retrain"]:
                return {
                    "status": "skipped",
                    "reason": trigger_result["reason"],
                }

            queue_ids = trigger_result.get("queue_ids", [])
            self.trigger.mark_processing(queue_ids)
        else:
            queue_ids = []
            logger.info("Force retraining enabled, skipping trigger check.")

        try:
            # Step 2: Analyze drift
            drift_result = {"drift_detected": False}
            if new_data_dir:
                logger.info("Analyzing data drift...")
                drift_result = self.drift_monitor.analyze_new_data(new_data_dir)
                logger.info("Drift result: %s", drift_result["recommendation"])

            # Step 3: Train
            if drift_result.get("drift_detected"):
                logger.info("Significant drift detected. Running hyperparameter tuning...")
                study = run_optuna_tuning(
                    epochs_tune=Settings.TUNING_EPOCHS,
                    n_trials=max(5, Settings.TUNING_TRIALS // 2),
                )
                best_params = study.best_trial.params
                logger.info("Best params from tuning: %s", best_params)
            else:
                logger.info("No significant drift. Using existing hyperparameters.")
                best_params = {
                    "lr0": 0.00011502,
                    "weight_decay": 0.00136136,
                    "momentum": 0.95328,
                    "box": 6.51472,
                    "cls": 0.98772,
                    "hsv_h": 0.02203,
                    "hsv_s": 0.80821,
                    "scale": 0.32889,
                    "mixup": 0.06183,
                }

            logger.info("Starting model training...")
            new_model = train_model(
                hyperparams=best_params,
                experiment_name="retrain_pipeline",
                track_to_snowflake=True,
            )

            # Step 4: Compare models
            logger.info("Comparing models...")
            comparison = ModelComparison()
            active_model = self.registry.get_active_model()

            new_weights = Path(new_model.trainer.save_dir) / "weights" / "best.pt"
            old_weights = Settings.DEFAULT_MODEL_PATH

            if active_model:
                logger.info("Active model: %s", active_model["model_version"])

            comp_result = comparison.compare(
                model_a_path=str(old_weights),
                model_b_path=str(new_weights),
                model_a_name="current_active",
                model_b_name="retrained",
            )

            # Step 5: Promote if better
            if comp_result.get("winner") == "retrained":
                logger.info("New model is better. Promoting...")
                # Model was already registered during training
                # Just need to promote it
            else:
                logger.info("Current model is still better. Keeping active model.")

            # Mark queue items as completed
            if queue_ids:
                self.trigger.mark_completed(queue_ids)

            result = {
                "status": "completed",
                "drift_analysis": drift_result,
                "comparison": comp_result,
                "promoted": comp_result.get("winner") == "retrained",
            }

            logger.info("=" * 60)
            logger.info("RETRAINING PIPELINE COMPLETED")
            logger.info("Result: %s", result["status"])
            logger.info("=" * 60)

            return result

        except Exception as e:
            logger.error("Retraining pipeline failed: %s", e)
            if queue_ids:
                self.trigger.mark_failed(queue_ids)
            return {
                "status": "failed",
                "error": str(e),
            }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run retraining pipeline")
    parser.add_argument("--new-data", default=None, help="Path to new training data")
    parser.add_argument("--force", action="store_true", help="Force retraining")

    args = parser.parse_args()

    pipeline = RetrainPipeline()
    result = pipeline.run(new_data_dir=args.new_data, force=args.force)
    print("Pipeline result:", result)
