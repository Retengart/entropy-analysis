"""Core analysis modules."""

from entropy_analysis.core.analyze import TextAnalyzer
from entropy_analysis.core.metrics import (
    calculate_compression_complexity,
    calculate_js_divergence,
    calculate_kl_divergence,
    calculate_mutual_information,
    calculate_simpson_index,
    calculate_zipf_coefficient,
)
from entropy_analysis.core.normalize import Alphabet, Normalizer
from entropy_analysis.core.stats import (
    bootstrap_entropy_confidence,
    calculate_extended_stats,
    calculate_miller_madow_correction,
    calculate_shannon_entropy,
)

__all__ = [
    "TextAnalyzer",
    "Normalizer",
    "Alphabet",
    "calculate_shannon_entropy",
    "calculate_extended_stats",
    "calculate_miller_madow_correction",
    "bootstrap_entropy_confidence",
    "calculate_kl_divergence",
    "calculate_js_divergence",
    "calculate_mutual_information",
    "calculate_simpson_index",
    "calculate_compression_complexity",
    "calculate_zipf_coefficient",
]
