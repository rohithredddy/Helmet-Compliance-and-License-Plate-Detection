"""
Main Streamlit application for Helmet Detection Pipeline.
Entry point with navigation and Snowflake connection status.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Helmet Detection Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E8EAED;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 1.0rem;
        color: #9AA0A6;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2a2a4a;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4FC3F7;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #9AA0A6;
        margin-top: 0.3rem;
    }

    .status-ok {
        color: #00CC96;
        font-weight: 600;
    }

    .status-error {
        color: #EF553B;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### Helmet Detection")
        st.markdown("Traffic Violation Detection System")
        st.divider()

        st.markdown("**Navigation**")
        st.page_link("app.py", label="Home")
        st.page_link("pages/1_EDA_Dashboard.py", label="EDA Dashboard")
        st.page_link("pages/2_Run_Inference.py", label="Run Inference")
        st.page_link("pages/3_Training_Metrics.py", label="Training Metrics")
        st.page_link("pages/4_Retrain_Model.py", label="Retrain Model")
        st.page_link("pages/5_Detection_History.py", label="Detection History")

        st.divider()

        # Snowflake connection status
        st.markdown("**Snowflake Status**")
        from components.snowflake_auth import test_connection
        connected, msg = test_connection()
        if connected:
            st.markdown(f'<span class="status-ok">Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-error">Disconnected</span>', unsafe_allow_html=True)
            st.caption(msg[:100])

    # Main content
    st.markdown('<div class="main-header">Helmet Detection Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Deep Learning Framework for Automated Helmet Violation Detection'
        '</div>',
        unsafe_allow_html=True,
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">96.3%</div>
            <div class="metric-label">Best mAP50</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">66.8%</div>
            <div class="metric-label">Best mAP50-95</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Detection Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">YOLOv8m</div>
            <div class="metric-label">Model Architecture</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Pipeline overview
    st.markdown("### Pipeline Overview")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Detection Classes")
        st.markdown("""
        | Class | Description |
        |-------|-------------|
        | Helmet | Rider wearing helmet |
        | NoHelmet | Rider without helmet |
        | Motorbike | Two-wheeler vehicle |
        | PNumber | Vehicle number plate |
        """)

        st.markdown("#### Violation Types")
        st.markdown("""
        - **No Helmet**: Rider detected without helmet (severity: high)
        """)

    with col_b:
        st.markdown("#### Pipeline Stages")
        st.markdown("""
        1. **Data Preparation** - Load, validate, and split YOLO datasets
        2. **Exploratory Data Analysis** - Class distributions, bbox stats, image quality
        3. **Model Training** - YOLOv8 with Optuna hyperparameter tuning
        4. **Inference** - Image and video detection with violation analysis
        5. **Retraining** - Automated pipeline triggered by new data
        """)

        st.markdown("#### Infrastructure")
        st.markdown("""
        - **Database**: Snowflake (HELMET_DETECTION_DB)
        - **Frontend**: Streamlit
        - **Model**: YOLOv8m (Ultralytics)
        - **Tuning**: Optuna
        """)


if __name__ == "__main__":
    main()
