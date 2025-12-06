"""
Statistical functions for entropy analysis.

Includes:
- Shannon entropy calculation
- Miller-Madow bias correction
- Bootstrap confidence intervals
- Extended statistics (quartiles, skewness, kurtosis)
- Outlier detection (IQR, Z-score, Modified Z-score)
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class ExtendedStats:
    """Extended statistical metrics for a distribution."""

    mean: float
    median: float
    std_dev: float
    variance: float
    q1: float
    q3: float
    iqr: float
    coefficient_of_variation: float
    skewness: float
    kurtosis: float
    min_value: float
    max_value: float


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis."""

    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int


@dataclass
class NormalizedEntropy:
    """Normalized entropy metrics."""

    h_normalized: float  # H / log2(alphabet_size), i.e. H / H_max
    entropy_per_word: float  # H / N (average entropy contribution per word)
    efficiency: float  # How close to maximum entropy (0-1), same as h_normalized


class OutlierMethod(Enum):
    """Methods for outlier detection."""

    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"


@dataclass
class OutlierResult:
    """Result of outlier detection for a single value."""

    is_outlier: bool
    score: float
    threshold: float
    method: OutlierMethod


def calculate_shannon_entropy(
    counts: Sequence[int] | NDArray[np.integer],
    base: float = 2.0,
) -> float:
    """
    Calculate Shannon entropy from counts.

    Args:
        counts: Array of counts for each category
        base: Logarithm base (default 2 for bits)

    Returns:
        Shannon entropy in the specified base
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    total = counts_arr.sum()

    if total == 0:
        return 0.0

    # Calculate probabilities, avoiding division by zero
    probs = counts_arr / total
    # Filter out zero probabilities (0 * log(0) = 0 by convention)
    probs = probs[probs > 0]

    # H = -sum(p * log(p))
    return float(-np.sum(probs * np.log(probs) / np.log(base)))


def calculate_shannon_entropy_from_probs(
    probs: Sequence[float] | NDArray[np.floating],
    base: float = 2.0,
) -> float:
    """
    Calculate Shannon entropy from probabilities.

    Args:
        probs: Array of probabilities (should sum to 1)
        base: Logarithm base (default 2 for bits)

    Returns:
        Shannon entropy in the specified base
    """
    probs_arr = np.asarray(probs, dtype=np.float64)
    # Filter out zero probabilities
    probs_arr = probs_arr[probs_arr > 0]

    if len(probs_arr) == 0:
        return 0.0

    return float(-np.sum(probs_arr * np.log(probs_arr) / np.log(base)))


def calculate_miller_madow_correction(
    entropy: float,
    n_categories: int,
    n_samples: int,
) -> float:
    """
    Apply Miller-Madow bias correction to entropy estimate.

    The correction compensates for the systematic underestimation
    of entropy from small samples.

    Args:
        entropy: Observed Shannon entropy
        n_categories: Number of non-zero categories (K)
        n_samples: Total number of samples (N)

    Returns:
        Bias-corrected entropy estimate
    """
    if n_samples == 0:
        return entropy

    # Miller-Madow correction: H_corrected = H_obs + (K - 1) / (2N)
    correction = (n_categories - 1) / (2 * n_samples)
    return entropy + correction


def calculate_normalized_entropy(
    entropy: float,
    alphabet_size: int,
    n_samples: int,
    base: float = 2.0,
) -> NormalizedEntropy:
    """
    Calculate normalized entropy metrics.

    Args:
        entropy: Shannon entropy value
        alphabet_size: Size of the alphabet (maximum categories)
        n_samples: Number of samples (words)
        base: Logarithm base

    Returns:
        NormalizedEntropy with various normalized metrics
    """
    max_entropy = np.log(alphabet_size) / np.log(base) if alphabet_size > 1 else 0.0

    h_normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    # Note: This is NOT the information-theoretic "entropy rate" (which is 
    # the limit of H(X_n|X_1...X_{n-1}) as n→∞). This is simply H/N,
    # representing average entropy contribution per word in this sample.
    entropy_per_word = entropy / n_samples if n_samples > 0 else 0.0
    efficiency = h_normalized  # Same as normalized entropy (0-1 scale)

    return NormalizedEntropy(
        h_normalized=float(h_normalized),
        entropy_per_word=float(entropy_per_word),
        efficiency=float(efficiency),
    )


def bootstrap_entropy_confidence(
    counts: Sequence[int] | NDArray[np.integer],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    base: float = 2.0,
    random_state: int | None = None,
) -> BootstrapResult:
    """
    Calculate bootstrap confidence interval for entropy.

    Uses vectorized operations for efficient computation. Much faster than
    parallelization via ProcessPoolExecutor due to numpy's optimized routines.

    Args:
        counts: Array of counts for each category
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (default 0.95 for 95% CI)
        base: Logarithm base
        random_state: Random seed for reproducibility

    Returns:
        BootstrapResult with estimate and confidence interval
    """
    counts_arr = np.asarray(counts, dtype=np.int64)
    total = int(counts_arr.sum())
    n_categories = len(counts_arr)

    if total == 0:
        return BootstrapResult(
            estimate=0.0,
            std_error=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        )

    rng = np.random.default_rng(random_state)

    # Original entropy
    original_entropy = calculate_shannon_entropy(counts_arr, base)

    # Convert counts to probabilities for multinomial sampling
    probs = counts_arr / total

    # Vectorized bootstrap using multinomial distribution
    # This generates all bootstrap samples at once: shape (n_bootstrap, n_categories)
    bootstrap_counts = rng.multinomial(total, probs, size=n_bootstrap)

    # Vectorized entropy calculation for all bootstrap samples
    # Avoid log(0) by masking zero counts
    bootstrap_probs = bootstrap_counts / total  # (n_bootstrap, n_categories)

    # Calculate entropy for each bootstrap sample
    # H = -sum(p * log(p)) for p > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        log_probs = np.log(bootstrap_probs) / np.log(base)
        log_probs = np.where(bootstrap_probs > 0, log_probs, 0.0)
        bootstrap_entropies = -np.sum(bootstrap_probs * log_probs, axis=1)

    std_error = float(np.std(bootstrap_entropies, ddof=1))

    # Percentile confidence interval
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_entropies, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_entropies, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        estimate=original_entropy,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def calculate_extended_stats(
    values: Sequence[float] | NDArray[np.floating],
) -> ExtendedStats | None:
    """
    Calculate extended statistical metrics.

    Args:
        values: Array of values to analyze

    Returns:
        ExtendedStats or None if insufficient data
    """
    arr = np.asarray(values, dtype=np.float64)

    if len(arr) < 2:
        return None

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std_dev = float(np.std(arr, ddof=1))
    variance = float(np.var(arr, ddof=1))

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    cv = (std_dev / abs(mean)) * 100 if abs(mean) > 1e-10 else 0.0

    # Skewness and kurtosis using scipy
    skewness = float(stats.skew(arr, bias=False))
    kurtosis = float(stats.kurtosis(arr, bias=False))  # Excess kurtosis

    return ExtendedStats(
        mean=mean,
        median=median,
        std_dev=std_dev,
        variance=variance,
        q1=q1,
        q3=q3,
        iqr=iqr,
        coefficient_of_variation=cv,
        skewness=skewness,
        kurtosis=kurtosis,
        min_value=float(np.min(arr)),
        max_value=float(np.max(arr)),
    )


def calculate_mad(values: Sequence[float] | NDArray[np.floating]) -> float:
    """
    Calculate Median Absolute Deviation (MAD).

    Args:
        values: Array of values

    Returns:
        MAD value
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0

    median = np.median(arr)
    deviations = np.abs(arr - median)
    return float(np.median(deviations))


def detect_outlier_iqr(
    value: float,
    q1: float,
    q3: float,
    iqr: float,
    k: float = 1.5,
) -> OutlierResult:
    """
    Detect if a value is an outlier using IQR method.

    Args:
        value: Value to test
        q1: First quartile
        q3: Third quartile
        iqr: Interquartile range
        k: Multiplier for IQR (default 1.5)

    Returns:
        OutlierResult
    """
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    is_outlier = value < lower_bound or value > upper_bound

    # Score: how many IQRs away from the nearest fence
    if iqr > 0:
        if value < lower_bound:
            score = (lower_bound - value) / iqr
        elif value > upper_bound:
            score = (value - upper_bound) / iqr
        else:
            score = 0.0
    else:
        score = 0.0

    return OutlierResult(
        is_outlier=is_outlier,
        score=score,
        threshold=k,
        method=OutlierMethod.IQR,
    )


def detect_outlier_zscore(
    value: float,
    mean: float,
    std_dev: float,
    threshold: float = 3.0,
) -> OutlierResult:
    """
    Detect if a value is an outlier using Z-score method.

    Args:
        value: Value to test
        mean: Distribution mean
        std_dev: Standard deviation
        threshold: Z-score threshold (default 3.0)

    Returns:
        OutlierResult
    """
    if std_dev < 1e-10:
        return OutlierResult(
            is_outlier=False,
            score=0.0,
            threshold=threshold,
            method=OutlierMethod.ZSCORE,
        )

    z_score = abs(value - mean) / std_dev

    return OutlierResult(
        is_outlier=z_score > threshold,
        score=z_score,
        threshold=threshold,
        method=OutlierMethod.ZSCORE,
    )


def detect_outlier_modified_zscore(
    value: float,
    median: float,
    mad: float,
    threshold: float = 3.5,
) -> OutlierResult:
    """
    Detect if a value is an outlier using Modified Z-score (MAD-based).

    Args:
        value: Value to test
        median: Distribution median
        mad: Median Absolute Deviation
        threshold: Modified Z-score threshold (default 3.5)

    Returns:
        OutlierResult
    """
    if mad < 1e-10:
        return OutlierResult(
            is_outlier=False,
            score=0.0,
            threshold=threshold,
            method=OutlierMethod.MODIFIED_ZSCORE,
        )

    # 0.6745 is the 75th percentile of the standard normal distribution
    modified_z = 0.6745 * abs(value - median) / mad

    return OutlierResult(
        is_outlier=modified_z > threshold,
        score=modified_z,
        threshold=threshold,
        method=OutlierMethod.MODIFIED_ZSCORE,
    )


def calculate_correlation(
    x: Sequence[float] | NDArray[np.floating],
    y: Sequence[float] | NDArray[np.floating],
) -> tuple[float, float, float, float, float] | None:
    """
    Calculate Pearson correlation and linear regression.

    Args:
        x: Independent variable values
        y: Dependent variable values

    Returns:
        Tuple of (correlation, slope, intercept, r_squared, p_value) or None
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if len(x_arr) < 2 or len(x_arr) != len(y_arr):
        return None

    # Calculate correlation using scipy
    correlation, p_value = stats.pearsonr(x_arr, y_arr)

    # Linear regression
    slope, intercept, _, _, _ = stats.linregress(x_arr, y_arr)

    r_squared = correlation**2

    return float(correlation), float(slope), float(intercept), float(r_squared), float(p_value)


# ============================================================================
# Enhanced Metrics for More Comprehensive Analysis
# ============================================================================


def calculate_perplexity(entropy: float, base: float = 2.0) -> float:
    """
    Calculate perplexity from entropy.

    Perplexity = base^H represents the number of equally likely outcomes
    that would produce the same entropy. For a distribution over k categories,
    if all categories were equally probable, the entropy would be log_base(k),
    so perplexity gives a measure of the "effective number of categories"
    being used by the distribution.

    Note: In language modeling, perplexity has a specific interpretation as
    the average branching factor. For letter distributions, it indicates
    how many letters are "effectively" contributing to the distribution.

    Args:
        entropy: Shannon entropy
        base: Base of entropy (default 2)

    Returns:
        Perplexity value (always >= 1)
    """
    return float(base**entropy)


def calculate_alphabet_utilization(
    n_unique: int,
    alphabet_size: int,
) -> float:
    """
    Calculate alphabet utilization - what fraction of alphabet is actually used.

    Args:
        n_unique: Number of unique letters actually used
        alphabet_size: Total alphabet size

    Returns:
        Utilization fraction (0-1)
    """
    if alphabet_size <= 0:
        return 0.0
    return float(n_unique / alphabet_size)


def calculate_herfindahl_index(
    counts: Sequence[int] | NDArray[np.integer],
) -> float:
    """
    Calculate Herfindahl-Hirschman Index (HHI) - measure of concentration.

    HHI = sum(p_i^2)

    A higher value indicates more concentration (less diversity).
    Range: [1/n, 1] where n is number of categories.

    Args:
        counts: Array of counts for each category

    Returns:
        Herfindahl index (0-1)
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    total = counts_arr.sum()

    if total == 0:
        return 0.0

    probs = counts_arr / total
    hhi = np.sum(probs**2)

    return float(hhi)


def calculate_uniformity_distance(
    counts: Sequence[int] | NDArray[np.integer],
    metric: str = "euclidean",
) -> float:
    """
    Calculate distance from uniform distribution.

    Args:
        counts: Array of counts for each category
        metric: Distance metric - "euclidean", "manhattan", or "chebyshev"

    Returns:
        Distance from uniform distribution
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    total = counts_arr.sum()

    if total == 0:
        return 0.0

    # Actual distribution
    probs = counts_arr / total

    # Uniform distribution
    n_categories = len(counts_arr)
    uniform = np.ones(n_categories) / n_categories

    # Calculate distance
    if metric == "euclidean":
        distance = np.sqrt(np.sum((probs - uniform) ** 2))
    elif metric == "manhattan":
        distance = np.sum(np.abs(probs - uniform))
    elif metric == "chebyshev":
        distance = np.max(np.abs(probs - uniform))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return float(distance)


def calculate_uniqueness_ratio(
    n_words: int,
    n_unique_letters: int,
) -> float:
    """
    Calculate uniqueness ratio - diversity per word.

    This represents how many unique letters are used per word on average.

    Args:
        n_words: Total number of words
        n_unique_letters: Number of unique letters used

    Returns:
        Uniqueness ratio
    """
    if n_words == 0:
        return 0.0
    return float(n_unique_letters / n_words)


def calculate_redundancy(
    entropy: float,
    max_entropy: float,
) -> float:
    """
    Calculate redundancy - how much information capacity is unused.

    R = 1 - H/H_max

    Args:
        entropy: Actual entropy
        max_entropy: Maximum possible entropy

    Returns:
        Redundancy (0-1, where 0 = no redundancy)
    """
    if max_entropy <= 0:
        return 0.0
    return float(1.0 - (entropy / max_entropy))


def calculate_evenness(
    counts: Sequence[int] | NDArray[np.integer],
) -> float:
    """
    Calculate Pielou's evenness index J.

    J = H / H_max = H / log(S)

    where S is the number of non-zero categories.
    Measures how evenly individuals are distributed across categories.

    Args:
        counts: Array of counts for each category

    Returns:
        Evenness index (0-1, where 1 = perfectly even)
    """
    counts_arr = np.asarray(counts, dtype=np.int64)
    n_nonzero = int(np.sum(counts_arr > 0))

    if n_nonzero <= 1:
        return 1.0

    h = calculate_shannon_entropy(counts_arr)
    h_max = float(np.log2(n_nonzero))

    if h_max == 0:
        return 0.0

    return float(h / h_max)


def calculate_renyi_entropy(
    counts: Sequence[int] | NDArray[np.integer],
    alpha: float,
    base: float = 2.0,
) -> float:
    """
    Calculate Rényi entropy of order alpha.

    H_α = (1/(1-α)) * log(sum(p_i^α))

    Special cases:
    - α → 0: H_0 = log(number of non-zero categories) - Hartley entropy
    - α → 1: H_1 = Shannon entropy
    - α = 2: H_2 = -log(sum(p_i^2)) - Collision entropy
    - α → ∞: H_∞ = -log(max(p_i)) - Min-entropy

    Args:
        counts: Array of counts for each category
        alpha: Order parameter (α)
        base: Logarithm base (default 2)

    Returns:
        Rényi entropy
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    total = counts_arr.sum()

    if total == 0:
        return 0.0

    probs = counts_arr / total
    probs = probs[probs > 0]

    if len(probs) == 0:
        return 0.0

    # Special cases
    if alpha == 0:
        # H_0 = log(number of non-zero categories)
        return float(np.log(len(probs)) / np.log(base))
    elif alpha == 1:
        # H_1 = Shannon entropy (limit as α → 1)
        return calculate_shannon_entropy_from_probs(probs, base)
    elif np.isinf(alpha):
        # H_∞ = min-entropy
        return float(-np.log(np.max(probs)) / np.log(base))
    else:
        # General case
        sum_p_alpha = np.sum(probs**alpha)
        renyi = (1 / (1 - alpha)) * np.log(sum_p_alpha) / np.log(base)
        return float(renyi)


@dataclass
class EnhancedMetrics:
    """Enhanced metrics for more comprehensive text analysis."""

    # Perplexity: 2^H, effective number of categories in use
    perplexity: float

    # Alphabet utilization: fraction of alphabet letters actually used
    alphabet_utilization: float  # 0-1

    # Concentration and diversity
    herfindahl_index: float  # Higher = more concentrated
    evenness: float  # Pielou's J, 0-1

    # Distance from ideal
    uniformity_distance: float  # Euclidean distance from uniform

    # Efficiency metrics
    uniqueness_ratio: float  # Unique letters per word
    redundancy: float  # 1 - H/H_max

    # Rényi entropies (different orders)
    renyi_0: float  # Hartley entropy (log of support)
    renyi_2: float  # Collision entropy
    renyi_inf: float  # Min-entropy


def calculate_enhanced_metrics(
    counts: NDArray[np.integer],
    n_words: int,
    alphabet_size: int,
    shannon_entropy: float,
) -> EnhancedMetrics:
    """
    Calculate all enhanced metrics at once.

    Args:
        counts: Array of letter counts
        n_words: Total number of words
        alphabet_size: Size of the alphabet
        shannon_entropy: Pre-calculated Shannon entropy

    Returns:
        EnhancedMetrics with all additional metrics
    """
    n_unique = int(np.sum(counts > 0))
    max_entropy = float(np.log2(alphabet_size))

    perplexity = calculate_perplexity(shannon_entropy)
    alphabet_util = calculate_alphabet_utilization(n_unique, alphabet_size)
    hhi = calculate_herfindahl_index(counts)
    evenness = calculate_evenness(counts)
    uniformity = calculate_uniformity_distance(counts, metric="euclidean")
    uniqueness = calculate_uniqueness_ratio(n_words, n_unique)
    redundancy = calculate_redundancy(shannon_entropy, max_entropy)

    renyi_0 = calculate_renyi_entropy(counts, alpha=0)
    renyi_2 = calculate_renyi_entropy(counts, alpha=2)
    renyi_inf = calculate_renyi_entropy(counts, alpha=float("inf"))

    return EnhancedMetrics(
        perplexity=perplexity,
        alphabet_utilization=alphabet_util,
        herfindahl_index=hhi,
        evenness=evenness,
        uniformity_distance=uniformity,
        uniqueness_ratio=uniqueness,
        redundancy=redundancy,
        renyi_0=renyi_0,
        renyi_2=renyi_2,
        renyi_inf=renyi_inf,
    )


# ============================================================================
# Advanced Statistical Tests for Maximum Objectivity
# ============================================================================


@dataclass
class PermutationTestResult:
    """Result of permutation test."""

    observed_difference: float
    p_value: float
    n_permutations: int
    is_significant: bool  # p < 0.05
    effect_direction: str  # "positive", "negative", or "none"


@dataclass
class EffectSizeResult:
    """Effect size measures (Cohen's d and others)."""

    cohens_d: float
    effect_magnitude: str  # "negligible", "small", "medium", "large"
    mean_difference: float
    pooled_std: float


@dataclass
class BootstrapDifferenceResult:
    """Bootstrap confidence interval for difference between two groups."""

    mean_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    is_significant: bool  # 0 not in CI
    n_bootstrap: int


@dataclass
class InformationDistances:
    """Advanced information-theoretic distances between distributions."""

    wasserstein: float  # Earth Mover's Distance
    hellinger: float  # Hellinger distance
    bhattacharyya: float  # Bhattacharyya distance
    total_variation: float  # Total variation distance


@dataclass
class StatisticalComparisonResult:
    """Complete statistical comparison of two groups."""

    # Basic statistics
    mean1: float
    mean2: float
    std1: float
    std2: float
    n1: int
    n2: int

    # Permutation test
    permutation_test: PermutationTestResult

    # Effect size
    effect_size: EffectSizeResult

    # Bootstrap CI for difference
    bootstrap_diff: BootstrapDifferenceResult

    # Classical tests
    t_test_p_value: float | None = None
    mann_whitney_p_value: float | None = None


def permutation_test(
    data1: Sequence[float] | NDArray[np.floating],
    data2: Sequence[float] | NDArray[np.floating],
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    random_state: int = 42,
) -> PermutationTestResult:
    """
    Permutation test for difference in means.

    Most robust test - no assumptions about distributions.

    Args:
        data1: First sample
        data2: Second sample
        n_permutations: Number of random permutations
        alternative: "two-sided", "greater", or "less"
        random_state: Random seed for reproducibility

    Returns:
        PermutationTestResult with p-value and interpretation
    """
    data1_arr = np.asarray(data1, dtype=np.float64)
    data2_arr = np.asarray(data2, dtype=np.float64)

    observed_diff = float(np.mean(data1_arr) - np.mean(data2_arr))

    # Combined data - make a copy to avoid modifying inputs
    combined = np.concatenate([data1_arr, data2_arr]).copy()
    n1 = len(data1_arr)

    # Generate permutation distribution
    rng = np.random.default_rng(random_state)
    null_distribution = np.empty(n_permutations)

    for i in range(n_permutations):
        rng.shuffle(combined)
        null_distribution[i] = np.mean(combined[:n1]) - np.mean(combined[n1:])

    # Calculate p-value
    if alternative == "two-sided":
        p_value = float(np.sum(np.abs(null_distribution) >= np.abs(observed_diff)) / n_permutations)
    elif alternative == "greater":
        p_value = float(np.sum(null_distribution >= observed_diff) / n_permutations)
    elif alternative == "less":
        p_value = float(np.sum(null_distribution <= observed_diff) / n_permutations)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Determine direction
    if observed_diff > 0:
        direction = "positive"
    elif observed_diff < 0:
        direction = "negative"
    else:
        direction = "none"

    return PermutationTestResult(
        observed_difference=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
        is_significant=p_value < 0.05,
        effect_direction=direction,
    )


def calculate_cohens_d(
    data1: Sequence[float] | NDArray[np.floating],
    data2: Sequence[float] | NDArray[np.floating],
) -> EffectSizeResult:
    """
    Calculate Cohen's d effect size.

    Measures the standardized difference between two means.

    Args:
        data1: First sample
        data2: Second sample

    Returns:
        EffectSizeResult with Cohen's d and interpretation
    """
    data1_arr = np.asarray(data1, dtype=np.float64)
    data2_arr = np.asarray(data2, dtype=np.float64)

    mean1 = float(np.mean(data1_arr))
    mean2 = float(np.mean(data2_arr))
    std1 = float(np.std(data1_arr, ddof=1))
    std2 = float(np.std(data2_arr, ddof=1))

    n1 = len(data1_arr)
    n2 = len(data2_arr)

    # Check for sufficient sample size for pooled std
    if n1 + n2 <= 2:
        return EffectSizeResult(
            cohens_d=0.0,
            effect_magnitude="undefined",
            mean_difference=mean1 - mean2,
            pooled_std=0.0,
        )

    # Pooled standard deviation
    pooled_std = float(np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)))

    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    # Interpret magnitude (Cohen's guidelines)
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return EffectSizeResult(
        cohens_d=float(cohens_d),
        effect_magnitude=magnitude,
        mean_difference=mean1 - mean2,
        pooled_std=pooled_std,
    )


def bootstrap_difference_ci(
    data1: Sequence[float] | NDArray[np.floating],
    data2: Sequence[float] | NDArray[np.floating],
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
) -> BootstrapDifferenceResult:
    """
    Bootstrap confidence interval for difference in means.

    Args:
        data1: First sample
        data2: Second sample
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95%)

    Returns:
        BootstrapDifferenceResult with CI for difference
    """
    data1_arr = np.asarray(data1, dtype=np.float64)
    data2_arr = np.asarray(data2, dtype=np.float64)

    observed_diff = float(np.mean(data1_arr) - np.mean(data2_arr))

    # Bootstrap
    rng = np.random.default_rng(42)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        sample1 = rng.choice(data1_arr, size=len(data1_arr), replace=True)
        sample2 = rng.choice(data2_arr, size=len(data2_arr), replace=True)
        diff = np.mean(sample1) - np.mean(sample2)
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Percentile CI
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))

    # Significant if 0 not in CI
    is_significant = not (ci_lower <= 0 <= ci_upper)

    return BootstrapDifferenceResult(
        mean_difference=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        is_significant=is_significant,
        n_bootstrap=n_bootstrap,
    )


def calculate_information_distances(
    dist1: Sequence[float] | NDArray[np.floating],
    dist2: Sequence[float] | NDArray[np.floating],
) -> InformationDistances:
    """
    Calculate advanced information-theoretic distances.

    Args:
        dist1: First probability distribution (PMF)
        dist2: Second probability distribution (PMF)

    Returns:
        InformationDistances with various distance metrics
    """
    p = np.asarray(dist1, dtype=np.float64)
    q = np.asarray(dist2, dtype=np.float64)

    # Normalize to ensure valid probability distributions
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q

    # Wasserstein distance (Earth Mover's Distance) for discrete distributions
    # For PMFs over categorical variables, we pass indices as values and probs as weights
    from scipy.stats import wasserstein_distance

    indices = np.arange(len(p))
    wasserstein = float(wasserstein_distance(indices, indices, p, q))

    # Hellinger distance: H(P,Q) = (1/√2) * ||√P - √Q||_2
    hellinger = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

    # Bhattacharyya distance: -ln(BC) where BC = Σ√(p_i * q_i)
    bc_coeff = np.sum(np.sqrt(p * q))
    bhattacharyya = float(-np.log(bc_coeff) if bc_coeff > 0 else float("inf"))

    # Total variation distance: TV(P,Q) = (1/2) * Σ|p_i - q_i|
    total_variation = float(0.5 * np.sum(np.abs(p - q)))

    return InformationDistances(
        wasserstein=wasserstein,
        hellinger=hellinger,
        bhattacharyya=bhattacharyya,
        total_variation=total_variation,
    )


def compare_groups_statistically(
    data1: Sequence[float] | NDArray[np.floating],
    data2: Sequence[float] | NDArray[np.floating],
    n_permutations: int = 10000,
    n_bootstrap: int = 5000,
) -> StatisticalComparisonResult:
    """
    Comprehensive statistical comparison of two groups.

    Combines multiple statistical tests for maximum objectivity.

    Args:
        data1: First sample
        data2: Second sample
        n_permutations: Number of permutations for permutation test
        n_bootstrap: Number of bootstrap samples

    Returns:
        StatisticalComparisonResult with all tests
    """
    data1_arr = np.asarray(data1, dtype=np.float64)
    data2_arr = np.asarray(data2, dtype=np.float64)

    # Basic statistics
    mean1 = float(np.mean(data1_arr))
    mean2 = float(np.mean(data2_arr))
    std1 = float(np.std(data1_arr, ddof=1))
    std2 = float(np.std(data2_arr, ddof=1))
    n1 = len(data1_arr)
    n2 = len(data2_arr)

    # Permutation test
    perm_result = permutation_test(data1_arr, data2_arr, n_permutations=n_permutations)

    # Effect size
    effect_result = calculate_cohens_d(data1_arr, data2_arr)

    # Bootstrap CI for difference
    bootstrap_result = bootstrap_difference_ci(data1_arr, data2_arr, n_bootstrap=n_bootstrap)

    # Classical tests (optional, for comparison)
    t_stat, t_p_value = stats.ttest_ind(data1_arr, data2_arr)
    u_stat, u_p_value = stats.mannwhitneyu(data1_arr, data2_arr, alternative="two-sided")

    return StatisticalComparisonResult(
        mean1=mean1,
        mean2=mean2,
        std1=std1,
        std2=std2,
        n1=n1,
        n2=n2,
        permutation_test=perm_result,
        effect_size=effect_result,
        bootstrap_diff=bootstrap_result,
        t_test_p_value=float(t_p_value),
        mann_whitney_p_value=float(u_p_value),
    )
