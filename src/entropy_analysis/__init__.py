"""
Entropy Analysis - Comprehensive text entropy analysis toolkit.

A modern Python toolkit for analyzing text entropy with:
- Shannon entropy and advanced metrics
- KL-divergence, Jensen-Shannon divergence for text comparison
- Bootstrap confidence intervals
- Interactive Plotly visualizations
- FastAPI backend and Streamlit dashboard
"""

__version__ = "2.0.0"
__author__ = "Chris"

from entropy_analysis.core.analyze import TextAnalyzer
from entropy_analysis.core.metrics import (
    calculate_js_divergence,
    calculate_kl_divergence,
    calculate_mutual_information,
    calculate_simpson_index,
)
from entropy_analysis.core.normalize import Alphabet, Normalizer
from entropy_analysis.core.stats import (
    bootstrap_entropy_confidence,
    calculate_miller_madow_correction,
    calculate_shannon_entropy,
)
from entropy_analysis.models.schemas import (
    ComparisonResponse,
    CorrelationResult,
    LetterDistribution,
    TextAnalysisResponse,
)

__all__ = [
    # Core
    "TextAnalyzer",
    "Normalizer",
    "Alphabet",
    # Metrics
    "calculate_kl_divergence",
    "calculate_js_divergence",
    "calculate_mutual_information",
    "calculate_simpson_index",
    # Stats
    "calculate_shannon_entropy",
    "calculate_miller_madow_correction",
    "bootstrap_entropy_confidence",
    # Models
    "TextAnalysisResponse",
    "LetterDistribution",
    "CorrelationResult",
    "ComparisonResponse",
]
