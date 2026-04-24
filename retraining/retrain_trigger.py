"""
Retraining trigger.
Checks the RETRAINING_QUEUE table for pending batches
and decides whether to trigger retraining.
"""

import logging
from config.settings import Settings
from config.snowflake_config import SnowflakeManager

logger = logging.getLogger(__name__)


class RetrainTrigger:
    """Check conditions and trigger retraining."""

    def __init__(self):
        self.sf = SnowflakeManager()
        self.min_images = Settings.MIN_RETRAIN_IMAGES

    def check_pending_batches(self):
        """Check for pending batches in the retraining queue."""
        df = self.sf.fetch_dataframe(
            """
            SELECT QUEUE_ID, BATCH_NAME, NUM_IMAGES, STAGE_PATH, REQUESTED_AT
            FROM RESULTS.RETRAINING_QUEUE
            WHERE STATUS = 'PENDING'
            ORDER BY REQUESTED_AT ASC
            """,
            schema="RESULTS",
        )
        return df

    def evaluate_trigger(self):
        """
        Evaluate whether retraining should be triggered.

        Returns:
            dict with decision, reason, and pending batch details
        """
        pending = self.check_pending_batches()

        if pending.empty:
            return {
                "should_retrain": False,
                "reason": "No pending batches in retraining queue.",
                "total_new_images": 0,
                "pending_batches": [],
            }

        total_images = int(pending["NUM_IMAGES"].sum())
        batch_names = pending["BATCH_NAME"].tolist()

        if total_images < self.min_images:
            return {
                "should_retrain": False,
                "reason": (
                    f"Only {total_images} new images available. "
                    f"Minimum required: {self.min_images}."
                ),
                "total_new_images": total_images,
                "pending_batches": batch_names,
            }

        return {
            "should_retrain": True,
            "reason": f"{total_images} new images ready for retraining.",
            "total_new_images": total_images,
            "pending_batches": batch_names,
            "queue_ids": pending["QUEUE_ID"].tolist(),
        }

    def add_to_queue(self, batch_name, num_images, stage_path=None, requested_by=None):
        """Add a new data batch to the retraining queue."""
        self.sf.insert_row(
            table="RETRAINING_QUEUE",
            data_dict={
                "BATCH_NAME": batch_name,
                "NUM_IMAGES": num_images,
                "STAGE_PATH": stage_path,
                "STATUS": "PENDING",
                "REQUESTED_BY": requested_by or "system",
            },
            schema="RESULTS",
        )
        logger.info("Added batch '%s' (%d images) to retraining queue.", batch_name, num_images)

    def mark_processing(self, queue_ids):
        """Mark batches as being processed."""
        for qid in queue_ids:
            self.sf.execute_query(
                "UPDATE RESULTS.RETRAINING_QUEUE SET STATUS = 'PROCESSING' WHERE QUEUE_ID = %s",
                schema="RESULTS",
                params=(qid,),
            )

    def mark_completed(self, queue_ids):
        """Mark batches as completed."""
        for qid in queue_ids:
            self.sf.execute_query(
                "UPDATE RESULTS.RETRAINING_QUEUE SET STATUS = 'COMPLETED', "
                "PROCESSED_AT = CURRENT_TIMESTAMP() WHERE QUEUE_ID = %s",
                schema="RESULTS",
                params=(qid,),
            )

    def mark_failed(self, queue_ids):
        """Mark batches as failed."""
        for qid in queue_ids:
            self.sf.execute_query(
                "UPDATE RESULTS.RETRAINING_QUEUE SET STATUS = 'FAILED' WHERE QUEUE_ID = %s",
                schema="RESULTS",
                params=(qid,),
            )
