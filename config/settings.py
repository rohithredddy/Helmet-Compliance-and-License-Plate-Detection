"""
Central configuration for the Helmet Detection project.
All paths, constants, class definitions, and model defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Project-wide settings and constants."""

    # -- Project root --
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # -- Class definitions --
    CLASS_NAMES = ["Helmet", "Motorbike", "NoHelmet", "PNumber"]
    NUM_CLASSES = 4
    CLASS_ID_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    # -- Color map for drawing (BGR for OpenCV) --
    COLOR_MAP = {
        "Helmet": (0, 255, 0),       # Green
        "Motorbike": (255, 0, 0),    # Blue
        "NoHelmet": (0, 0, 255),     # Red
        "PNumber": (245, 66, 188),   # Pink
    }

    # -- Color map for Streamlit/Plotly (RGB hex) --
    COLOR_MAP_HEX = {
        "Helmet": "#00FF00",
        "Motorbike": "#0000FF",
        "NoHelmet": "#FF0000",
        "PNumber": "#F542BC",
    }

    # -- Dataset paths --
    DATA_DIR = PROJECT_ROOT / "Data"
    DATA_YAML = DATA_DIR / "data.yaml"
    IMAGES_DIR = DATA_DIR / "images"
    LABELS_DIR = DATA_DIR / "labels"
    SPLITS = ["train", "val", "test"]

    # -- Model defaults --
    DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "weights" / "best.pt"
    BASE_MODEL = "yolov8m.pt"
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 960))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))
    BATCH_SIZE = 8

    # -- Training defaults --
    DEFAULT_EPOCHS = 120
    TUNING_EPOCHS = 30
    TUNING_TRIALS = 15
    PATIENCE = 20

    # -- Inference data --
    INFERENCE_DATA_DIR = PROJECT_ROOT / "Inference Data"
    INFERENCE_RESULTS_DIR = PROJECT_ROOT / "Inference_results"

    # -- Results --
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_CSV = RESULTS_DIR / "results.csv"
    WEIGHTS_DIR = RESULTS_DIR / "weights"

    # -- Supported file extensions --
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

    # -- Violation thresholds --
    PROXIMITY_PIXELS = 150       # pixel distance to associate rider with bike

    # -- Snowflake object names --
    SF_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "HELMET_DETECTION_DB")
    SF_SCHEMA_RAW = "RAW_DATA"
    SF_SCHEMA_PROCESSED = "PROCESSED"
    SF_SCHEMA_MODELS = "MODELS"
    SF_SCHEMA_RESULTS = "RESULTS"

    SF_STAGE_RAW_IMAGES = "RAW_IMAGES_STAGE"
    SF_STAGE_MODEL_WEIGHTS = "MODEL_WEIGHTS_STAGE"
    SF_STAGE_INFERENCE_UPLOADS = "INFERENCE_UPLOADS_STAGE"
    SF_STAGE_TRAINING_DATA = "TRAINING_DATA_STAGE"

    # -- Minimum new data for retraining --
    MIN_RETRAIN_IMAGES = 50
