"""Core analysis modules."""

from entropy_analysis.core.analyze import TextAnalyzer
from entropy_analysis.core.metrics import (
    # Divergence metrics
    calculate_compression_complexity,
    calculate_js_divergence,
    calculate_kl_divergence,
    calculate_mutual_information,
    calculate_simpson_index,
    calculate_zipf_coefficient,
    # Unified pipeline
    calculate_all_metrics,
    extract_attribution_features,
    calculate_attribution_distance,
    quick_attribution_score,
    # Result types
    ComprehensiveMetrics,
    AttributionFeatures,
)
from entropy_analysis.core.normalize import Alphabet, Normalizer
from entropy_analysis.core.stats import (
    bootstrap_entropy_confidence,
    calculate_extended_stats,
    calculate_miller_madow_correction,
    calculate_shannon_entropy,
    correct_multiple_comparisons,
    compare_groups_statistically,
)

__all__ = [
    # Main classes
    "TextAnalyzer",
    "Normalizer",
    "Alphabet",
    # Stats
    "calculate_shannon_entropy",
    "calculate_extended_stats",
    "calculate_miller_madow_correction",
    "bootstrap_entropy_confidence",
    "correct_multiple_comparisons",
    "compare_groups_statistically",
    # Divergence
    "calculate_kl_divergence",
    "calculate_js_divergence",
    "calculate_mutual_information",
    "calculate_simpson_index",
    "calculate_compression_complexity",
    "calculate_zipf_coefficient",
    # Unified pipeline
    "calculate_all_metrics",
    "extract_attribution_features",
    "calculate_attribution_distance",
    "quick_attribution_score",
    "ComprehensiveMetrics",
    "AttributionFeatures",
]
