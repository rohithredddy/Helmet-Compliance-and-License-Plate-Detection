import torch
import optuna
from ultralytics import YOLO
from optuna.pruners import MedianPruner


def run_optuna_tuning(
    data_yaml,
    base_model="yolov8m.pt",
    epochs_tune=30,
    img_size=960,
    batch_size=8,
    n_trials=15,
    device=None,
):
    """
    Runs Optuna hyperparameter tuning for YOLOv8.
    Returns study object.
    """

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

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
                **params
            )

            metrics = model.val()
            map5095 = metrics.box.map

            trial.report(map5095, step=0)
            return map5095

        except Exception as e:
            print("Trial failed:", e)
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_warmup_steps=5)
    )

    study.optimize(objective, n_trials=n_trials)

    print("\nBest Trial:")
    print("mAP50-95:", study.best_trial.value)
    print("Best Params:", study.best_trial.params)

    return study


def run_final_training(
    data_yaml,
    best_params,
    base_model="yolov8m.pt",
    epochs_final=120,
    img_size=960,
    batch_size=8,
    experiment_name="optuna_final",
    device=None,
):
    """
    Runs final training using best Optuna parameters.
    """

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    final_model = YOLO(base_model)

    final_model.train(
        data=data_yaml,
        epochs=epochs_final,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        optimizer="AdamW",
        cos_lr=True,
        patience=20,
        amp=True,
        name=experiment_name,
        plots=True,
        **best_params
    )

    print("Final Training Complete.")

    return final_model


# Optional CLI usage
if __name__ == "__main__":

    DATA_YAML = "Dataset_helmet_resplit/data.yaml"

    study = run_optuna_tuning(
        data_yaml=DATA_YAML,
        base_model="yolov8m.pt"
    )

    run_final_training(
        data_yaml=DATA_YAML,
        best_params=study.best_trial.params
    )