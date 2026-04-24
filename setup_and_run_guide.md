# Setup & Run Guide: Helmet Violation Detection

This guide will walk you through cloning the repository, setting up the required environment, configuring Snowflake, and running the Streamlit dashboard and inference pipelines.

## Prerequisites
Before you begin, ensure you have the following installed on your system:
*   **Git**
*   **Python 3.8 to 3.11** (Ensure `pip` is available)
*   **A Snowflake Account** (You will need your account URL, username, password, and role)

---

## 1. Clone the Repository

Open your terminal or command prompt and run the following commands to clone the project to your local machine:

```bash
# Clone the repository
git clone https://github.com/rohithredddy/Final-Year-Project.git

# Navigate into the project directory
cd Final-Year-Project
```

---

## 2. Environment Setup

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.

### Create and Activate Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

Once the virtual environment is activated, install the required packages:
```bash
pip install -r requirements.txt
```

---

## 3. Configure Snowflake Credentials

The project relies heavily on Snowflake for metadata, model weights, and inference tracking.

1.  Copy the environment template file:
    ```bash
    cp .env.example .env
    ```
    *(On Windows Command Prompt, use `copy .env.example .env`)*

2.  Open the newly created `.env` file in your preferred text editor.
3.  Fill in your Snowflake connection details:
    ```ini
    SNOWFLAKE_ACCOUNT=your_account_identifier (e.g., xy12345.us-east-1)
    SNOWFLAKE_USER=your_username
    SNOWFLAKE_PASSWORD=your_password
    SNOWFLAKE_ROLE=your_role (e.g., ACCOUNTADMIN or SYSADMIN)
    SNOWFLAKE_DATABASE=HELMET_DETECTION_DB
    ```

---

## 4. Provision Snowflake Infrastructure

Run the automated setup script to generate the required database, schemas, stages, and tables in your Snowflake account.

```bash
python snowflake_setup/run_setup.py
```

*Expected output: The script will connect to Snowflake and execute `01_create_database.sql`, `02_create_stages.sql`, and `03_create_tables.sql` sequentially.*

---

## 5. Prepare Data & Register Model

To ensure the Streamlit dashboard displays the correct analytics, you must load the initial data and register the provided pre-trained model.

### Upload Data & Run EDA
```bash
# 1. Upload dataset metadata to Snowflake
python data_preparation/upload_to_snowflake.py

# 2. Validate dataset integrity
python data_preparation/data_validation.py

# 3. Generate and upload EDA statistics
python eda/generate_eda_report.py
```

### Register the Existing YOLOv8 Model
The repository comes with a pre-trained `best.pt` file (mAP50=96.3%). Register it in Snowflake so the inference engine can fetch it:

```bash
python -c "from training.model_registry import ModelRegistry; ModelRegistry().register_existing_model()"
```

---

## 6. Launch the Streamlit Web Application

The interactive dashboard allows you to view EDA insights, monitor training, and run live inference on new images or videos.

```bash
# Navigate to the streamlit_app directory
cd streamlit_app

# Start the Streamlit server
streamlit run app.py
```

*   The terminal will provide a local URL, typically `http://localhost:8501`.
*   Open this URL in your web browser to access the system.

---

## 7. Running Inference via CLI (Optional)

If you prefer to bypass the UI and run inference directly from the command line:

### Batch Inference on a Folder of Images
```bash
python inference/batch_inference.py --input "Inference Data/real imgs" --output batch_output
```

### Inference on a Single Image
```bash
python -c "
from inference.detector import HelmetDetector
detector = HelmetDetector()
result = detector.detect_image('path/to/your/image.jpg')
print(f'Detections: {result[\"num_detections\"]}, Violations: {result[\"num_violations\"]}')
"
```

## Troubleshooting
*   **Snowflake Connection Errors:** Ensure your IP address is not blocked by your Snowflake Network Policy. Double-check the `SNOWFLAKE_ACCOUNT` format in the `.env` file.
*   **CUDA Out of Memory:** If you are running locally with a GPU, the inference script will auto-detect it. If it fails due to memory, reduce `BATCH_SIZE` in `config/settings.py` or run inference on CPU.
