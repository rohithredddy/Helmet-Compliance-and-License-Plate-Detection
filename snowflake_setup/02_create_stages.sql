-- ============================================================
-- Helmet Detection Project - Internal Stages Setup
-- ============================================================

USE DATABASE HELMET_DETECTION_DB;

-- Stage for raw training images metadata/configs
CREATE STAGE IF NOT EXISTS RAW_DATA.RAW_IMAGES_STAGE
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Stores raw dataset configuration files and image metadata';

-- Stage for trained model weight files (.pt)
CREATE STAGE IF NOT EXISTS MODELS.MODEL_WEIGHTS_STAGE
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Stores serialized model weight files';

-- Stage for user-uploaded inference images/videos
CREATE STAGE IF NOT EXISTS RESULTS.INFERENCE_UPLOADS_STAGE
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Stores images and videos uploaded for inference';

-- Stage for new training data batches (retraining pipeline)
CREATE STAGE IF NOT EXISTS RAW_DATA.TRAINING_DATA_STAGE
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Stores new data batches for retraining';
