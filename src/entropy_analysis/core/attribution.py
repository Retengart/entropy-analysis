"""
Authorship attribution metrics based on Delta family methods.

Includes:
- Burrows' Delta (Z-score standardized Manhattan distance)
- Eder's Delta (Weighted Z-scores)
- Cosine Delta (Z-score standardized Cosine distance)
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cityblock, cosine


@dataclass
class DeltaResult:
    """Result of Delta calculation."""

    delta: float
    method: str
    n_features: int  # Number of MFW used


def calculate_z_scores(
    features: NDArray[np.floating],
    means: NDArray[np.floating],
    stds: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Calculate Z-scores for features.

    Z = (x - mu) / sigma

    Args:
        features: Feature vector (frequencies)
        means: Mean values for each feature across corpus
        stds: Standard deviations for each feature across corpus

    Returns:
        Z-score vector
    """
    # Avoid division by zero
    stds_safe = np.where(stds == 0, 1.0, stds)
    return (features - means) / stds_safe


def calculate_burrows_delta(
    target_vector: NDArray[np.floating],
    candidate_vector: NDArray[np.floating],
) -> float:
    """
    Calculate Burrows' Delta (Manhattan distance of Z-scores).

    Delta = (1/n) * sum(|Z_target - Z_candidate|)

    Args:
        target_vector: Z-score vector of target text
        candidate_vector: Z-score vector of candidate text (or centroid)

    Returns:
        Delta value
    """
    return float(cityblock(target_vector, candidate_vector) / len(target_vector))


def calculate_eders_delta(
    target_vector: NDArray[np.floating],
    candidate_vector: NDArray[np.floating],
) -> float:
    """
    Calculate Eder's Delta (Weighted Manhattan distance).

    Eder's Delta gives more weight to more frequent words (higher rank).
    Weights decrease linearly with rank.

    Args:
        target_vector: Z-score vector of target text
        candidate_vector: Z-score vector of candidate text

    Returns:
        Eder's Delta value
    """
    n = len(target_vector)
    
    # Weights: decreasing from 1.0 down to 0.0
    # w_i = 1 - (rank / n)
    # Since vectors are usually sorted by frequency (MFW), rank i is i+1
    ranks = np.arange(1, n + 1)
    weights = 1.0 - (ranks / (n + 1))  # Avoid zero weight at the end
    
    # Weighted sum of absolute differences
    diffs = np.abs(target_vector - candidate_vector)
    weighted_diffs = diffs * weights
    
    return float(np.sum(weighted_diffs))


def calculate_cosine_delta(
    target_vector: NDArray[np.floating],
    candidate_vector: NDArray[np.floating],
) -> float:
    """
    Calculate Cosine Delta (Cosine distance of Z-scores).

    Cosine Delta = 1 - cos(Z_target, Z_candidate)
    
    Considered state-of-the-art for authorship attribution on many languages.
    It effectively normalizes the vector lengths.

    Args:
        target_vector: Z-score vector of target text
        candidate_vector: Z-score vector of candidate text

    Returns:
        Cosine Delta value (0-2, usually 0-1 for positive vectors)
    """
    # scipy's cosine returns distance (1 - similarity)
    return float(cosine(target_vector, candidate_vector))
