"""
Snowflake connection manager.
Provides a reusable session/connection factory for all modules.
"""

import os
import logging
from contextlib import contextmanager
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

logger = logging.getLogger(__name__)


class SnowflakeManager:
    """Manages Snowflake connections and provides utility methods."""

    def __init__(self):
        try:
            import streamlit as st
            self.account = os.getenv("SNOWFLAKE_ACCOUNT") or st.secrets.get("SNOWFLAKE_ACCOUNT")
            self.user = os.getenv("SNOWFLAKE_USER") or st.secrets.get("SNOWFLAKE_USER")
            self.password = os.getenv("SNOWFLAKE_PASSWORD") or st.secrets.get("SNOWFLAKE_PASSWORD")
            self.role = os.getenv("SNOWFLAKE_ROLE") or st.secrets.get("SNOWFLAKE_ROLE")
            self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE") or st.secrets.get("SNOWFLAKE_WAREHOUSE", "HELMET_WH")
            self.database = os.getenv("SNOWFLAKE_DATABASE") or st.secrets.get("SNOWFLAKE_DATABASE", "HELMET_DETECTION_DB")
        except Exception:
            self.account = os.getenv("SNOWFLAKE_ACCOUNT")
            self.user = os.getenv("SNOWFLAKE_USER")
            self.password = os.getenv("SNOWFLAKE_PASSWORD")
            self.role = os.getenv("SNOWFLAKE_ROLE")
            self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "HELMET_WH")
            self.database = os.getenv("SNOWFLAKE_DATABASE", "HELMET_DETECTION_DB")

        if not all([self.account, self.user, self.password]):
            raise EnvironmentError(
                "Missing Snowflake credentials. "
                "Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD in .env"
            )

    def _get_connection_params(self, schema=None):
        """Build connection parameter dict."""
        params = {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "role": self.role,
            "warehouse": self.warehouse,
            "database": self.database,
        }
        if schema:
            params["schema"] = schema
        return params

    def get_connection(self, schema=None):
        """
        Create and return a raw snowflake.connector connection.
        Caller is responsible for closing.
        """
        params = self._get_connection_params(schema)
        logger.info("Connecting to Snowflake account: %s", self.account)
        return snowflake.connector.connect(**params)

    @contextmanager
    def connection(self, schema=None):
        """Context manager for auto-closing connections."""
        conn = self.get_connection(schema)
        try:
            yield conn
        finally:
            conn.close()
            logger.info("Snowflake connection closed.")

    def execute_query(self, query, schema=None, params=None):
        """Execute a single query and return results."""
        with self.connection(schema) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return columns, rows
                return None, None
            finally:
                cursor.close()

    def execute_queries(self, queries, schema=None):
        """Execute multiple queries in sequence."""
        with self.connection(schema) as conn:
            cursor = conn.cursor()
            try:
                for query in queries:
                    query = query.strip()
                    if query:
                        logger.info("Executing: %s", query[:80])
                        cursor.execute(query)
                conn.commit()
                logger.info("All queries executed successfully.")
            finally:
                cursor.close()

    def execute_sql_file(self, filepath):
        """Read and execute all statements from a SQL file."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on semicolons, filter empty statements
        statements = [s.strip() for s in content.split(";") if s.strip()]
        self.execute_queries(statements)

    def upload_file_to_stage(self, local_path, stage_name, schema=None, auto_compress=False, overwrite=True):
        """Upload a local file to a Snowflake internal stage."""
        with self.connection(schema) as conn:
            cursor = conn.cursor()
            try:
                put_cmd = (
                    f"PUT 'file://{local_path}' @{stage_name} "
                    f"AUTO_COMPRESS={'TRUE' if auto_compress else 'FALSE'} "
                    f"OVERWRITE={'TRUE' if overwrite else 'FALSE'}"
                )
                cursor.execute(put_cmd)
                result = cursor.fetchall()
                logger.info("Upload result: %s", result)
                return result
            finally:
                cursor.close()

    def download_file_from_stage(self, stage_path, local_dir, schema=None):
        """Download a file from a Snowflake internal stage."""
        with self.connection(schema) as conn:
            cursor = conn.cursor()
            try:
                get_cmd = f"GET '{stage_path}' 'file://{local_dir}'"
                cursor.execute(get_cmd)
                result = cursor.fetchall()
                logger.info("Download result: %s", result)
                return result
            finally:
                cursor.close()

    def insert_row(self, table, data_dict, schema=None):
        """Insert a single row into a table from a dict."""
        columns = ", ".join(data_dict.keys())
        placeholders = ", ".join(["%s"] * len(data_dict))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        values = list(data_dict.values())

        with self.connection(schema) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, values)
                conn.commit()
            finally:
                cursor.close()

    def insert_dataframe(self, table, df, schema=None):
        """Insert a pandas DataFrame into a table."""
        if df.empty:
            logger.warning("Empty DataFrame, skipping insert.")
            return

        columns = ", ".join(df.columns)
        placeholders = ", ".join(["%s"] * len(df.columns))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        with self.connection(schema) as conn:
            cursor = conn.cursor()
            try:
                rows = [tuple(row) for row in df.values]
                cursor.executemany(query, rows)
                conn.commit()
                logger.info("Inserted %d rows into %s", len(rows), table)
            finally:
                cursor.close()

    def fetch_dataframe(self, query, schema=None):
        """Execute query and return results as a pandas DataFrame."""
        import pandas as pd

        columns, rows = self.execute_query(query, schema)
        if columns and rows:
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame()
