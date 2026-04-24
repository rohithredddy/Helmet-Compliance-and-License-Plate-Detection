"""
Result logger for persisting inference results to Snowflake.
Logs both inference metadata and individual detections.
"""

import uuid
import logging

from config.snowflake_config import SnowflakeManager

logger = logging.getLogger(__name__)


class ResultLogger:
    """Log inference results to Snowflake RESULTS schema."""

    def __init__(self):
        self.sf = SnowflakeManager()

    def log_inference(self, image_name, result, source_type="upload", model_version=None):
        """
        Log a complete inference result to Snowflake.

        Args:
            image_name: name of the processed image/video
            result: dict from HelmetDetector.detect_image()
            source_type: 'upload', 'batch', or 'video'
            model_version: version string of the model used

        Returns:
            inference_id
        """
        inference_id = f"inf_{uuid.uuid4().hex[:12]}"

        # Log inference metadata
        self.sf.insert_row(
            table="INFERENCE_LOGS",
            data_dict={
                "INFERENCE_ID": inference_id,
                "IMAGE_NAME": image_name,
                "SOURCE_TYPE": source_type,
                "MODEL_VERSION": model_version,
                "NUM_DETECTIONS": result.get("num_detections", 0),
                "NUM_VIOLATIONS": result.get("num_violations", 0),
                "PROCESSING_TIME_MS": result.get("processing_time_ms", 0),
            },
            schema="RESULTS",
        )

        # Log individual detections
        detections = result.get("detections", [])
        violations = result.get("violations", [])

        # Build a map of detection index -> violation type
        violation_map = {}
        for v in violations:
            for idx in v.get("detection_indices", []):
                violation_map[idx] = v["type"]

        for i, det in enumerate(detections):
            bbox = det.get("bbox", {})
            self.sf.insert_row(
                table="DETECTION_RESULTS",
                data_dict={
                    "INFERENCE_ID": inference_id,
                    "CLASS_NAME": det["class_name"],
                    "CLASS_ID": det["class_id"],
                    "CONFIDENCE": det["confidence"],
                    "BBOX_X1": bbox.get("x1"),
                    "BBOX_Y1": bbox.get("y1"),
                    "BBOX_X2": bbox.get("x2"),
                    "BBOX_Y2": bbox.get("y2"),
                    "VIOLATION_TYPE": violation_map.get(i),
                },
                schema="RESULTS",
            )

        logger.info(
            "Logged inference %s: %d detections, %d violations",
            inference_id, len(detections), len(violations),
        )
        return inference_id

    def get_recent_inferences(self, limit=50):
        """Fetch recent inference logs."""
        return self.sf.fetch_dataframe(
            f"SELECT * FROM RESULTS.INFERENCE_LOGS ORDER BY CREATED_AT DESC LIMIT {limit}",
            schema="RESULTS",
        )

    def get_detection_details(self, inference_id):
        """Fetch detection details for a specific inference."""
        return self.sf.fetch_dataframe(
            f"SELECT * FROM RESULTS.DETECTION_RESULTS WHERE INFERENCE_ID = '{inference_id}'",
            schema="RESULTS",
        )

    def get_violation_stats(self, days=30):
        """Get violation statistics for the last N days."""
        return self.sf.fetch_dataframe(
            f"""
            SELECT
                d.VIOLATION_TYPE,
                COUNT(*) as COUNT,
                AVG(d.CONFIDENCE) as AVG_CONFIDENCE,
                DATE(i.CREATED_AT) as DATE
            FROM RESULTS.DETECTION_RESULTS d
            JOIN RESULTS.INFERENCE_LOGS i ON d.INFERENCE_ID = i.INFERENCE_ID
            WHERE d.VIOLATION_TYPE IS NOT NULL
                AND i.CREATED_AT >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
            GROUP BY d.VIOLATION_TYPE, DATE(i.CREATED_AT)
            ORDER BY DATE DESC
            """,
            schema="RESULTS",
        )
