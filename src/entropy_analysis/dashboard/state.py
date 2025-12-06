"""
Session state management for the dashboard.
"""

import streamlit as st

from entropy_analysis.core.analyze import TextAnalyzer


def init_session_state():
    """Initialize session state variables."""
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = TextAnalyzer.create()
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []
    if "comparison_texts" not in st.session_state:
        st.session_state.comparison_texts = {}



