"""
Snowflake setup runner.
Executes all SQL scripts in order to provision the database infrastructure.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.snowflake_config import SnowflakeManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent


def run_setup():
    """Execute all SQL setup scripts in order."""
    sql_files = sorted(SQL_DIR.glob("*.sql"))

    if not sql_files:
        logger.error("No SQL files found in %s", SQL_DIR)
        return False

    sf = SnowflakeManager()

    for sql_file in sql_files:
        logger.info("=" * 60)
        logger.info("Executing: %s", sql_file.name)
        logger.info("=" * 60)
        try:
            sf.execute_sql_file(str(sql_file))
            logger.info("Successfully executed: %s", sql_file.name)
        except Exception as e:
            logger.error("Failed to execute %s: %s", sql_file.name, e)
            return False

    logger.info("=" * 60)
    logger.info("All Snowflake setup scripts executed successfully.")
    logger.info("=" * 60)

    # Verify by listing schemas
    logger.info("Verifying setup...")
    try:
        columns, rows = sf.execute_query(
            "SELECT SCHEMA_NAME FROM HELMET_DETECTION_DB.INFORMATION_SCHEMA.SCHEMATA"
        )
        if rows:
            logger.info("Schemas found: %s", [r[0] for r in rows])

        columns, rows = sf.execute_query(
            "SELECT TABLE_SCHEMA, TABLE_NAME FROM HELMET_DETECTION_DB.INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA')"
        )
        if rows:
            for schema, table in rows:
                logger.info("  Table: %s.%s", schema, table)

        logger.info("Setup verification complete.")
    except Exception as e:
        logger.warning("Verification query failed (non-critical): %s", e)

    return True


if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)
