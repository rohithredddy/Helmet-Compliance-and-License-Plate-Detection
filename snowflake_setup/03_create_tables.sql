-- ============================================================
-- Helmet Detection Project - Tables Setup
-- ============================================================

USE DATABASE HELMET_DETECTION_DB;

-- -------------------------------------------------------
-- RAW_DATA schema tables
-- -------------------------------------------------------

CREATE TABLE IF NOT EXISTS RAW_DATA.DATASET_METADATA (
    RECORD_ID           NUMBER AUTOINCREMENT PRIMARY KEY,
    IMAGE_FILENAME      VARCHAR(500) NOT NULL,
    SPLIT               VARCHAR(10) NOT NULL,       -- train, val, test
    IMAGE_WIDTH         NUMBER,
    IMAGE_HEIGHT        NUMBER,
    FILE_SIZE_BYTES     NUMBER,
    NUM_OBJECTS         NUMBER DEFAULT 0,
    HELMET_COUNT        NUMBER DEFAULT 0,
    NOHELMET_COUNT      NUMBER DEFAULT 0,
    MOTORBIKE_COUNT     NUMBER DEFAULT 0,
    PNUMBER_COUNT       NUMBER DEFAULT 0,
    CREATED_AT          TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- -------------------------------------------------------
-- PROCESSED schema tables
-- -------------------------------------------------------

CREATE TABLE IF NOT EXISTS PROCESSED.EDA_STATISTICS (
    STAT_ID             NUMBER AUTOINCREMENT PRIMARY KEY,
    CATEGORY            VARCHAR(100) NOT NULL,       -- class_distribution, bbox_stats, image_quality
    METRIC_NAME         VARCHAR(200) NOT NULL,
    METRIC_VALUE        FLOAT,
    SPLIT               VARCHAR(10),
    DETAILS             VARIANT,                     -- JSON for complex stats
    GENERATED_AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- -------------------------------------------------------
-- MODELS schema tables
-- -------------------------------------------------------

CREATE TABLE IF NOT EXISTS MODELS.TRAINING_RUNS (
    RUN_ID              VARCHAR(50) PRIMARY KEY,
    MODEL_BASE          VARCHAR(100) NOT NULL,
    HYPERPARAMETERS     VARIANT NOT NULL,            -- JSON of all hyperparams
    EPOCHS_PLANNED      NUMBER,
    EPOCHS_COMPLETED    NUMBER,
    BEST_MAP50          FLOAT,
    BEST_MAP50_95       FLOAT,
    BEST_PRECISION      FLOAT,
    BEST_RECALL         FLOAT,
    TRAINING_TIME_SEC   FLOAT,
    STATUS              VARCHAR(20) DEFAULT 'RUNNING',  -- RUNNING, COMPLETED, FAILED
    STARTED_AT          TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    COMPLETED_AT        TIMESTAMP_NTZ
);

CREATE TABLE IF NOT EXISTS MODELS.TRAINING_METRICS (
    METRIC_ID           NUMBER AUTOINCREMENT PRIMARY KEY,
    RUN_ID              VARCHAR(50) NOT NULL REFERENCES MODELS.TRAINING_RUNS(RUN_ID),
    EPOCH               NUMBER NOT NULL,
    TRAIN_BOX_LOSS      FLOAT,
    TRAIN_CLS_LOSS      FLOAT,
    TRAIN_DFL_LOSS      FLOAT,
    VAL_BOX_LOSS        FLOAT,
    VAL_CLS_LOSS        FLOAT,
    VAL_DFL_LOSS        FLOAT,
    PRECISION_B         FLOAT,
    RECALL_B            FLOAT,
    MAP50               FLOAT,
    MAP50_95            FLOAT,
    LEARNING_RATE       FLOAT,
    EPOCH_TIME_SEC      FLOAT,
    RECORDED_AT         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS MODELS.MODEL_REGISTRY (
    MODEL_ID            NUMBER AUTOINCREMENT PRIMARY KEY,
    MODEL_VERSION       VARCHAR(50) NOT NULL UNIQUE,
    RUN_ID              VARCHAR(50) REFERENCES MODELS.TRAINING_RUNS(RUN_ID),
    STAGE_PATH          VARCHAR(500) NOT NULL,
    MODEL_BASE          VARCHAR(100),
    MAP50               FLOAT,
    MAP50_95            FLOAT,
    PRECISION_VAL       FLOAT,
    RECALL_VAL          FLOAT,
    FILE_SIZE_BYTES     NUMBER,
    IS_ACTIVE           BOOLEAN DEFAULT FALSE,
    REGISTERED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    NOTES               VARCHAR(1000)
);

-- -------------------------------------------------------
-- RESULTS schema tables
-- -------------------------------------------------------

CREATE TABLE IF NOT EXISTS RESULTS.INFERENCE_LOGS (
    INFERENCE_ID        VARCHAR(50) PRIMARY KEY,
    IMAGE_NAME          VARCHAR(500),
    SOURCE_TYPE         VARCHAR(20) NOT NULL,        -- upload, batch, video
    MODEL_VERSION       VARCHAR(50),
    NUM_DETECTIONS      NUMBER DEFAULT 0,
    NUM_VIOLATIONS      NUMBER DEFAULT 0,
    PROCESSING_TIME_MS  FLOAT,
    CREATED_AT          TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS RESULTS.DETECTION_RESULTS (
    DETECTION_ID        NUMBER AUTOINCREMENT PRIMARY KEY,
    INFERENCE_ID        VARCHAR(50) NOT NULL REFERENCES RESULTS.INFERENCE_LOGS(INFERENCE_ID),
    CLASS_NAME          VARCHAR(50) NOT NULL,
    CLASS_ID            NUMBER NOT NULL,
    CONFIDENCE          FLOAT NOT NULL,
    BBOX_X1             FLOAT,
    BBOX_Y1             FLOAT,
    BBOX_X2             FLOAT,
    BBOX_Y2             FLOAT,
    VIOLATION_TYPE      VARCHAR(50),                 -- no_helmet, null
    CREATED_AT          TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS RESULTS.RETRAINING_QUEUE (
    QUEUE_ID            NUMBER AUTOINCREMENT PRIMARY KEY,
    BATCH_NAME          VARCHAR(200) NOT NULL,
    NUM_IMAGES          NUMBER NOT NULL,
    STAGE_PATH          VARCHAR(500),
    STATUS              VARCHAR(20) DEFAULT 'PENDING',  -- PENDING, PROCESSING, COMPLETED, FAILED
    REQUESTED_BY        VARCHAR(100),
    REQUESTED_AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PROCESSED_AT        TIMESTAMP_NTZ
);
