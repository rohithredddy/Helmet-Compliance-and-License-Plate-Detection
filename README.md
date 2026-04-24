# Helmet Violation Detection Pipeline

A production-grade deep learning framework for automated helmet violation detection,
built with YOLOv8 and deployed on Snowflake with Streamlit.

---

## Architecture

```
Data Preparation --> EDA --> Training --> Inference --> Monitoring
      |                        |             |             |
      v                        v             v             v
  Snowflake              Snowflake      Snowflake     Snowflake
  RAW_DATA               MODELS         RESULTS       RESULTS
```

### Detection Classes

| Class | ID | Description |
|-------|----|-------------|
| Helmet | 0 | Rider wearing helmet |
| Motorbike | 1 | Two-wheeler vehicle |
| NoHelmet | 2 | Rider without helmet |
| PNumber | 3 | Vehicle number plate |

### Violation Types

- **No Helmet** (severity: high) - Rider detected without helmet

---

## Project Structure

```
Project1/
├── config/                    # Central configuration and Snowflake connection
│   ├── settings.py            # All project constants, paths, and defaults
│   └── snowflake_config.py    # Snowflake connection manager
│
├── snowflake_setup/           # Database infrastructure (SQL + runner)
│   ├── 01_create_database.sql # Database, schemas, warehouse
│   ├── 02_create_stages.sql   # Internal stages for files
│   ├── 03_create_tables.sql   # All 7 tables DDL
│   └── run_setup.py           # Execute all SQL scripts
│
├── data_preparation/          # Data loading, splitting, validation
│   ├── dataset_loader.py      # Parse YOLO-format datasets
│   ├── data_splitter.py       # Train/val/test split logic
│   ├── upload_to_snowflake.py # Upload metadata to Snowflake
│   └── data_validation.py     # Integrity checks
│
├── eda/                       # Exploratory data analysis
│   ├── class_distribution.py  # Class counts and imbalance
│   ├── bbox_analysis.py       # Bounding box statistics
│   ├── image_quality.py       # Resolution, brightness analysis
│   └── generate_eda_report.py # Run all EDA and upload results
│
├── training/                  # Model training pipeline
│   ├── train.py               # Training with Snowflake tracking
│   ├── hyperparameter_tuning.py # Optuna-based tuning
│   ├── experiment_tracker.py  # Log runs/epochs to Snowflake
│   └── model_registry.py      # Version and promote models
│
├── inference/                 # Detection and violation analysis
│   ├── detector.py            # YOLOv8 inference engine
│   ├── violation_logic.py     # No-helmet logic
│   ├── result_logger.py       # Log results to Snowflake
│   └── batch_inference.py     # Process folders of images/videos
│
├── retraining/                # Automated retraining pipeline
│   ├── data_drift_monitor.py  # Distribution drift detection
│   ├── retrain_trigger.py     # Queue management and conditions
│   ├── retrain_pipeline.py    # End-to-end orchestrator
│   └── model_comparison.py    # Compare model versions
│
├── streamlit_app/             # Web application
│   ├── app.py                 # Main entry point
│   ├── components/            # Reusable UI components
│   └── pages/                 # 5 dashboard pages
│
├── Data/                      # Training dataset (YOLO format)
├── results/                   # Trained model weights and metrics
├── notebooks/                 # Reference Jupyter notebooks
└── requirements.txt           # Python dependencies
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/rohithredddy/Final-Year-Project.git
cd Final-Year-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Snowflake

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your Snowflake credentials
# SNOWFLAKE_ACCOUNT=your_account
# SNOWFLAKE_USER=your_username
# SNOWFLAKE_PASSWORD=your_password
# SNOWFLAKE_ROLE=your_role
```

### 3. Provision Snowflake Infrastructure

```bash
python snowflake_setup/run_setup.py
```

This creates:
- Database: `HELMET_DETECTION_DB`
- 4 Schemas: `RAW_DATA`, `PROCESSED`, `MODELS`, `RESULTS`
- 4 Internal Stages
- 7 Tables

### 4. Load Data and Run EDA

```bash
# Upload dataset metadata to Snowflake
python data_preparation/upload_to_snowflake.py

# Validate dataset integrity
python data_preparation/data_validation.py

# Generate and upload EDA statistics
python eda/generate_eda_report.py
```

### 5. Register Existing Model

The trained model (best.pt, mAP50=96.3%) is already available.
Register it in Snowflake:

```bash
python -c "
from training.model_registry import ModelRegistry
registry = ModelRegistry()
registry.register_existing_model()
"
```

### 6. Launch Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`

---

## Pipeline Commands

### Run Inference

> **Note**: For best results, the pipeline automatically resizes inputs to 960x960 during inference to match the training process.

```bash
# Single image
python -c "
from inference.detector import HelmetDetector
detector = HelmetDetector()
result = detector.detect_image('path/to/image.jpg')
print(f'Detections: {result[\"num_detections\"]}, Violations: {result[\"num_violations\"]}')
"

# Batch processing
python inference/batch_inference.py --input "Inference Data/real imgs" --output batch_output

# Video processing
python -c "
from inference.detector import HelmetDetector
detector = HelmetDetector()
results = detector.detect_video('path/to/video.mp4', output_path='output.mp4')
"
```

### Retrain Model

```bash
# Check if retraining is needed
python -c "
from retraining.retrain_trigger import RetrainTrigger
trigger = RetrainTrigger()
print(trigger.evaluate_trigger())
"

# Run retraining pipeline
python retraining/retrain_pipeline.py --new-data path/to/new_data --force
```

---

## Snowflake Infrastructure

### Database Schema

| Schema | Purpose |
|--------|---------|
| RAW_DATA | Raw dataset metadata and configs |
| PROCESSED | EDA statistics and transformed data |
| MODELS | Training runs, metrics, model registry |
| RESULTS | Inference logs, detections, retraining queue |

### Tables

| Table | Schema | Description |
|-------|--------|-------------|
| DATASET_METADATA | RAW_DATA | Image-level metadata and class counts |
| EDA_STATISTICS | PROCESSED | Computed EDA metrics |
| TRAINING_RUNS | MODELS | Training run metadata |
| TRAINING_METRICS | MODELS | Per-epoch training metrics |
| MODEL_REGISTRY | MODELS | Model version management |
| INFERENCE_LOGS | RESULTS | Inference session logs |
| DETECTION_RESULTS | RESULTS | Individual detection records |
| RETRAINING_QUEUE | RESULTS | New data batch queue |

---

## Model Performance

| Metric | Value |
|--------|-------|
| mAP50 | 96.3% |
| mAP50-95 | 66.8% |
| Precision | 94.5% |
| Recall | 92.9% |
| Architecture | YOLOv8m |
| Image Size | 960px |
| Optimizer | AdamW |
| Tuning | Optuna (15 trials) |

---

## Dataset

Source: [Helmet Detection Dataset - Roboflow Universe](https://universe.roboflow.com/hele/helmet-detection-nsbwm-ftr3v/dataset/4)

- Format: YOLO
- Split: train / val / test
- Includes day and night traffic images

---

## License

MIT License
