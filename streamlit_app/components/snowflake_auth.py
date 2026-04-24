"""
Snowflake authentication helper for Streamlit app.
Provides cached connection management.
"""

import streamlit as st
from config.snowflake_config import SnowflakeManager


@st.cache_resource
def get_snowflake_manager():
    """Get a cached SnowflakeManager instance."""
    try:
        return SnowflakeManager()
    except Exception as e:
        st.error(f"Failed to initialize Snowflake connection: {e}")
        return None


def require_snowflake():
    """
    Ensure Snowflake is connected.
    Shows error and stops if connection fails.
    Returns SnowflakeManager instance.
    """
    sf = get_snowflake_manager()
    if sf is None:
        st.error(
            "Snowflake connection not available. "
            "Please check your .env file and ensure credentials are set."
        )
        st.stop()
    return sf


def test_connection():
    """Test the Snowflake connection and return status."""
    try:
        sf = SnowflakeManager()
        columns, rows = sf.execute_query("SELECT CURRENT_TIMESTAMP()")
        return True, f"Connected at {rows[0][0]}"
    except Exception as e:
        return False, str(e)
