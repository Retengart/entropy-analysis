"""
Advanced metrics for text analysis and comparison.

Includes:
- Kullback-Leibler divergence
- Jensen-Shannon divergence
- Mutual Information
- Simpson's Index (Index of Coincidence)
- Lempel-Ziv complexity
- Zipf's law analysis
- Rolling/sliding window entropy
- N-gram entropy (conditional)
- Lexical diversity (Yule's K, MTLD)
- Burstiness
"""

import gzip
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special, sparse, stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon

from entropy_analysis.core.stats import calculate_shannon_entropy_from_probs


@dataclass
class DivergenceResult:
    """Result of divergence calculation between two distributions."""

    divergence: float
    symmetric: bool
    metric_name: str


@dataclass
class ZipfAnalysis:
    """Result of Zipf's law analysis."""

    alpha: float  # Zipf exponent
    r_squared: float  # Goodness of fit (R² from log-log regression)
    chi2_statistic: float  # Chi-squared statistic for goodness-of-fit
    chi2_pvalue: float  # Chi-squared test p-value (higher = better fit)


@dataclass
class RollingEntropyResult:
    """Result of rolling/sliding window entropy analysis."""

    positions: list[int]  # Window start positions
    entropies: list[float]  # Entropy values at each position
    window_size: int
    mean_entropy: float
    std_entropy: float
    min_entropy: float
    max_entropy: float


@dataclass
class ComplexityResult:
    """Result of complexity analysis."""

    raw_size: int
    compressed_size: int
    compression_ratio: float
    normalized_complexity: float


def normalize_to_probabilities(
    counts: Sequence[int] | NDArray[np.integer],
) -> NDArray[np.floating]:
    """Convert counts to probability distribution."""
    counts_arr = np.asarray(counts, dtype=np.float64)
    total = counts_arr.sum()
    if total == 0:
        return np.zeros_like(counts_arr, dtype=np.float64)
    return counts_arr / total


def _entropy_from_counts(counts_arr: NDArray[np.integer]) -> float:
    """
    Stable entropy from counts using scipy.special.xlogy (handles zeros safely).
    """
    counts_f = np.asarray(counts_arr, dtype=np.float64)
    total = counts_f.sum()
    if total == 0:
        return 0.0
    probs = counts_f / total
    # xlogy returns 0 when prob == 0, avoiding -inf * 0 issues
    return float(-np.sum(special.xlogy(probs, probs) / np.log(2)))


def calculate_kl_divergence(
    p: Sequence[float] | NDArray[np.floating],
    q: Sequence[float] | NDArray[np.floating],
    base: float = 2.0,
    epsilon: float = 1e-10,
) -> DivergenceResult:
    """
    Calculate Kullback-Leibler divergence D_KL(P || Q).

    Measures how much distribution P differs from reference Q.
    Note: KL divergence is asymmetric and undefined if q_i=0 where p_i>0.

    Args:
        p: Probability distribution P (should sum to 1)
        q: Reference distribution Q (should sum to 1)
        base: Logarithm base (default 2 for bits)
        epsilon: Small value added to Q only to avoid log(0)

    Returns:
        DivergenceResult with KL divergence value
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    # Normalize inputs to ensure they are valid probability distributions
    p_sum = p_arr.sum()
    q_sum = q_arr.sum()
    if p_sum > 0:
        p_arr = p_arr / p_sum
    if q_sum > 0:
        q_arr = q_arr / q_sum

    # Add epsilon only to Q (reference distribution) to handle q_i=0
    # This is standard practice: we're asking "how surprised would Q be by P?"
    # If Q assigns 0 probability to something P has, that's infinite surprise.
    # Adding epsilon to Q gives a large but finite value instead.
    q_smoothed = q_arr + epsilon
    q_smoothed = q_smoothed / q_smoothed.sum()  # Renormalize Q only

    # D_KL(P || Q) = sum(P * log(P/Q)) for all i where p_i > 0
    # Use xlogy for numerical stability: xlogy(p, p/q) = p * log(p/q)
    # By convention, 0 * log(0/q) = 0
    mask = p_arr > 0
    if np.any(mask):
        ratio = p_arr[mask] / q_smoothed[mask]
        kl_div = np.sum(special.xlogy(p_arr[mask], ratio) / np.log(base))
    else:
        kl_div = 0.0

    return DivergenceResult(
        divergence=float(kl_div),
        symmetric=False,
        metric_name="Kullback-Leibler Divergence",
    )


def calculate_js_divergence(
    p: Sequence[float] | NDArray[np.floating],
    q: Sequence[float] | NDArray[np.floating],
    base: float = 2.0,
) -> DivergenceResult:
    """
    Calculate Jensen-Shannon divergence.

    JSD is a symmetric and smoothed version of KL divergence.
    JSD = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M), where M = (P + Q) / 2

    Args:
        p: First probability distribution
        q: Second probability distribution
        base: Logarithm base (default 2 for bits)

    Returns:
        DivergenceResult with JS divergence value
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    # Use scipy's implementation
    js_div = jensenshannon(p_arr, q_arr, base=base)

    return DivergenceResult(
        divergence=float(js_div**2),  # scipy returns sqrt(JSD)
        symmetric=True,
        metric_name="Jensen-Shannon Divergence",
    )


def calculate_mutual_information(
    x: Sequence[int] | NDArray[np.integer],
    y: Sequence[int] | NDArray[np.integer],
) -> float:
    """
    Calculate mutual information between two categorical variables.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)

    where H is Shannon entropy.

    Args:
        x: First categorical variable (as integer labels)
        y: Second categorical variable (as integer labels)

    Returns:
        Mutual information value in bits
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if len(x_arr) != len(y_arr) or len(x_arr) == 0:
        return 0.0

    n = len(x_arr)

    # Count marginal frequencies
    x_counts = Counter(x_arr.tolist())
    y_counts = Counter(y_arr.tolist())

    # Count joint frequencies
    joint_counts = Counter(zip(x_arr.tolist(), y_arr.tolist()))

    # Calculate H(X)
    x_probs = np.array(list(x_counts.values())) / n
    h_x = -np.sum(x_probs * np.log2(x_probs))

    # Calculate H(Y)
    y_probs = np.array(list(y_counts.values())) / n
    h_y = -np.sum(y_probs * np.log2(y_probs))

    # Calculate H(X,Y)
    joint_probs = np.array(list(joint_counts.values())) / n
    h_xy = -np.sum(joint_probs * np.log2(joint_probs))

    # MI(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = h_x + h_y - h_xy

    return float(max(0.0, mi))  # Clamp to non-negative due to numerical errors


def calculate_simpson_index(
    counts: Sequence[int] | NDArray[np.integer],
) -> float:
    """
    Calculate Simpson's Index (Index of Coincidence).

    The probability that two randomly chosen items are the same.
    D = sum(n_i * (n_i - 1)) / (N * (N - 1))

    A lower value indicates higher diversity.

    Args:
        counts: Array of counts for each category

    Returns:
        Simpson's Index (0-1, where 1 = all same category)
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    n = counts_arr.sum()

    if n <= 1:
        return 1.0

    # D = sum(n_i * (n_i - 1)) / (N * (N - 1))
    simpson = np.sum(counts_arr * (counts_arr - 1)) / (n * (n - 1))

    return float(simpson)


def calculate_gini_simpson_index(
    counts: Sequence[int] | NDArray[np.integer],
) -> float:
    """
    Calculate Gini-Simpson Index (1 - Simpson's Index).

    The probability that two randomly chosen items are different.
    A higher value indicates higher diversity.

    Args:
        counts: Array of counts for each category

    Returns:
        Gini-Simpson Index (0-1, where 1 = maximum diversity)
    """
    return 1.0 - calculate_simpson_index(counts)


def calculate_compression_complexity(
    text: str,
    encoding: str = "utf-8",
) -> ComplexityResult:
    """
    Calculate compression-based complexity (approximation to Kolmogorov complexity).

    Uses gzip compression to estimate the algorithmic complexity of text.

    Args:
        text: Input text
        encoding: Text encoding

    Returns:
        ComplexityResult with compression metrics
    """
    text_bytes = text.encode(encoding)
    raw_size = len(text_bytes)

    if raw_size == 0:
        return ComplexityResult(
            raw_size=0,
            compressed_size=0,
            compression_ratio=0.0,
            normalized_complexity=0.0,
        )

    compressed = gzip.compress(text_bytes, compresslevel=9)
    compressed_size = len(compressed)

    compression_ratio = compressed_size / raw_size
    normalized_complexity = min(1.0, compression_ratio)

    return ComplexityResult(
        raw_size=raw_size,
        compressed_size=compressed_size,
        compression_ratio=compression_ratio,
        normalized_complexity=normalized_complexity,
    )


def calculate_lempel_ziv_complexity(text: str) -> int:
    """
    Calculate Lempel-Ziv complexity (number of distinct phrases).

    Based on the LZ76 algorithm - counts the number of unique substrings
    encountered when parsing left to right. Each new phrase extends the
    longest previously seen prefix by exactly one character.

    Args:
        text: Input text (typically converted to symbol sequence)

    Returns:
        Number of distinct phrases (complexity measure)
    """
    if not text:
        return 0

    n = len(text)
    vocabulary: set[str] = set()
    i = 0

    while i < n:
        # Find the longest prefix starting at position i that's in vocabulary
        j = i + 1
        while j <= n and text[i:j] in vocabulary:
            j += 1

        # text[i:j] is the shortest new phrase (longest known prefix + 1 char)
        # or text[i:j] where j > n means we've exhausted the string
        if j <= n:
            vocabulary.add(text[i:j])

        i = j

    # Number of distinct phrases equals vocabulary size
    # (each iteration adds exactly one phrase until we exhaust the string)
    return len(vocabulary) if vocabulary else 1


def calculate_zipf_coefficient(
    counts: Sequence[int] | NDArray[np.integer],
) -> ZipfAnalysis:
    """
    Analyze how well the distribution follows Zipf's law.

    Zipf's law: frequency ~ 1/rank^alpha

    Args:
        counts: Array of counts (will be sorted by frequency)

    Returns:
        ZipfAnalysis with exponent and goodness of fit
    """
    counts_arr = np.asarray(counts, dtype=np.float64)

    # Sort by frequency (descending)
    sorted_counts = np.sort(counts_arr)[::-1]

    # Remove zeros
    sorted_counts = sorted_counts[sorted_counts > 0]

    if len(sorted_counts) < 3:
        return ZipfAnalysis(
            alpha=0.0,
            r_squared=0.0,
            chi2_statistic=float("inf"),
            chi2_pvalue=0.0,
        )

    # Ranks (1, 2, 3, ...)
    ranks = np.arange(1, len(sorted_counts) + 1, dtype=np.float64)

    # Log-log regression: log(freq) = -alpha * log(rank) + const
    log_ranks = np.log(ranks)
    log_counts = np.log(sorted_counts)

    slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_counts)

    alpha = -slope  # Zipf exponent

    # Calculate expected frequencies based on fitted Zipf
    expected_log = intercept + slope * log_ranks
    expected_counts = np.exp(expected_log)

    # Scale expected counts to match total observed count for chi-squared test
    total_observed = sorted_counts.sum()
    expected_scaled = expected_counts * (total_observed / expected_counts.sum())

    # Chi-squared goodness-of-fit test (more appropriate for discrete distributions)
    # Combine bins with expected < 5 to meet chi-squared assumptions
    min_expected = 5.0
    if np.any(expected_scaled < min_expected):
        # Pool small categories for valid chi-squared
        mask = expected_scaled >= min_expected
        if np.sum(mask) >= 2:
            obs_pooled = np.append(sorted_counts[mask], sorted_counts[~mask].sum())
            exp_pooled = np.append(expected_scaled[mask], expected_scaled[~mask].sum())
        else:
            # Not enough categories, use all data
            obs_pooled = sorted_counts
            exp_pooled = expected_scaled
    else:
        obs_pooled = sorted_counts
        exp_pooled = expected_scaled

    # Chi-squared test
    # Note: df = n_categories - 1 - n_parameters_estimated (alpha and intercept = 2)
    chi2_stat = np.sum((obs_pooled - exp_pooled) ** 2 / exp_pooled)
    df = max(1, len(obs_pooled) - 1 - 2)  # -2 for estimated alpha and intercept
    chi2_pvalue = float(1 - stats.chi2.cdf(chi2_stat, df))

    return ZipfAnalysis(
        alpha=float(alpha),
        r_squared=float(r_value**2),
        chi2_statistic=float(chi2_stat),
        chi2_pvalue=float(chi2_pvalue),
    )


def calculate_rolling_entropy(
    sequence: Sequence[str],
    window_size: int,
    step_size: int = 1,
    alphabet: list[str] | None = None,
) -> RollingEntropyResult:
    """
    Calculate rolling/sliding window entropy through a sequence.

    Useful for detecting changes in text structure (e.g., style changes).

    Args:
        sequence: Sequence of tokens/characters
        window_size: Size of the sliding window
        step_size: Step size between windows
        alphabet: Optional fixed alphabet for counting

    Returns:
        RollingEntropyResult with entropy values at each position
    """
    seq_list = list(sequence)
    n = len(seq_list)

    if n < window_size:
        return RollingEntropyResult(
            positions=[],
            entropies=[],
            window_size=window_size,
            mean_entropy=0.0,
            std_entropy=0.0,
            min_entropy=0.0,
            max_entropy=0.0,
        )

    positions: list[int] = []
    entropies: list[float] = []

    # Build alphabet if not provided
    if alphabet is None:
        alphabet = list(set(seq_list))
    alphabet_index = {char: i for i, char in enumerate(alphabet)}

    for start in range(0, n - window_size + 1, step_size):
        window = seq_list[start : start + window_size]

        # Count occurrences
        counts = np.zeros(len(alphabet), dtype=np.int64)
        for char in window:
            if char in alphabet_index:
                counts[alphabet_index[char]] += 1

        # Calculate entropy
        probs = normalize_to_probabilities(counts)
        h = calculate_shannon_entropy_from_probs(probs)

        positions.append(start)
        entropies.append(h)

    entropies_arr = np.array(entropies) if entropies else np.array([0.0])

    return RollingEntropyResult(
        positions=positions,
        entropies=entropies,
        window_size=window_size,
        mean_entropy=float(np.mean(entropies_arr)),
        std_entropy=float(np.std(entropies_arr)),
        min_entropy=float(np.min(entropies_arr)),
        max_entropy=float(np.max(entropies_arr)),
    )


def calculate_conditional_entropy(
    joint_counts: NDArray[np.integer],
) -> float:
    """
    Calculate conditional entropy H(Y|X) from joint distribution.

    H(Y|X) = H(X,Y) - H(X)

    Args:
        joint_counts: 2D array of joint counts [x, y]

    Returns:
        Conditional entropy in bits
    """
    joint_arr = np.asarray(joint_counts, dtype=np.float64)
    total = joint_arr.sum()

    if total == 0:
        return 0.0

    # Marginal distribution of X
    x_counts = joint_arr.sum(axis=1)

    # H(X)
    h_x = calculate_shannon_entropy_from_probs(normalize_to_probabilities(x_counts.astype(int)))

    # H(X,Y) - joint entropy
    joint_probs = joint_arr.flatten() / total
    h_xy = calculate_shannon_entropy_from_probs(joint_probs)

    # H(Y|X) = H(X,Y) - H(X)
    return float(h_xy - h_x)


# Note: calculate_redundancy is defined in stats.py to avoid duplication
# Import it from there if needed: from entropy_analysis.core.stats import calculate_redundancy


def calculate_yules_k(tokens: Sequence[str]) -> float:
    """
    Calculate Yule's K (lexical diversity / concentration).

    Yule's K measures vocabulary richness. Higher K = more repetition (less diverse).

    Formula: K = 10⁴ × (S₂ - S₁) / S₁²
    where:
    - S₁ = N (total number of tokens)
    - S₂ = Σ(fᵢ²) (sum of squared frequencies for each unique word type)

    Args:
        tokens: List of tokens/words

    Returns:
        Yule's K value (typically 0-200, higher = more repetitive)
    """
    N = len(tokens)
    if N == 0:
        return 0.0

    counts = Counter(tokens)

    # S1 = total tokens
    S1 = N

    # S2 = sum of squared frequencies for each unique word type
    S2 = sum(freq**2 for freq in counts.values())

    # K = 10000 * (S2 - S1) / S1^2
    K = 10000 * (S2 - S1) / (S1**2)
    return float(K)


def calculate_mtld(tokens: Sequence[str], threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity (MTLD).
    Calculates average length of segment that maintains TTR >= threshold.

    MTLD counts how many times the TTR drops below threshold (a "factor"),
    then divides total tokens by number of factors. Higher MTLD = more diverse.

    Args:
        tokens: List of tokens/words
        threshold: TTR threshold (default 0.72)

    Returns:
        MTLD value (higher = more lexically diverse)
    """

    def mtld_pass(input_tokens: Sequence[str]) -> float:
        factors = 0.0
        current_tokens: set[str] = set()
        token_count = 0

        for token in input_tokens:
            current_tokens.add(token)
            token_count += 1
            ttr = len(current_tokens) / token_count

            if ttr < threshold:
                factors += 1
                current_tokens = set()
                token_count = 0

        # Partial factor for remaining text that didn't complete a full factor
        # The partial factor represents how far toward the threshold we got
        # If TTR is still 1.0 (all unique), partial = 0 (no progress toward factor)
        # If TTR is at threshold, partial = 1.0 (full factor)
        if token_count > 0:
            ttr = len(current_tokens) / token_count
            # Clamp TTR to avoid division issues and handle edge cases
            # partial_factor = (1 - TTR) / (1 - threshold)
            # When TTR = 1.0: partial = 0 (maximally diverse, no factor contribution)
            # When TTR = threshold: partial = 1.0 (full factor)
            partial = (1.0 - ttr) / (1.0 - threshold)
            factors += max(0.0, min(1.0, partial))

        if factors == 0:
            # Text is so diverse that TTR never dropped below threshold
            # Return the text length as MTLD (infinite diversity approximation)
            return float(len(input_tokens))

        return len(input_tokens) / factors

    if not tokens:
        return 0.0

    forward = mtld_pass(tokens)
    backward = mtld_pass(tokens[::-1])

    # Average of forward and backward passes
    return (forward + backward) / 2.0


def calculate_ngram_entropy(tokens: Sequence[str], n: int = 2) -> float:
    """
    Calculate N-gram conditional entropy.
    H(Xn | Xn-1 ... X1) = H(Xn ... X1) - H(Xn-1 ... X1)

    For Bigrams (n=2): H(X2 | X1) = H(X1, X2) - H(X1)

    Args:
        tokens: List of tokens/words (or characters)
        n: N-gram order

    Returns:
        Conditional entropy in bits
    """
    if len(tokens) < n:
        return 0.0
    
    # For large corpora, use Counter (memory-efficient for sparse data)
    # CSR optimization would require mapping n-grams to indices, which is complex
    # Counter is already sparse (only stores observed n-grams) and efficient
    # For very large datasets, consider pre-filtering or sampling
    ngrams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    context = Counter(tuple(tokens[i : i + n - 1]) for i in range(len(tokens) - n + 1))

    # Use stable entropy calculation with xlogy
    h_ngram = _entropy_from_counts(np.fromiter(ngrams.values(), dtype=np.int64))
    h_context = _entropy_from_counts(np.fromiter(context.values(), dtype=np.int64))

    # Clamp to non-negative to mitigate numerical drift
    return float(max(0.0, h_ngram - h_context))


def calculate_lexical_richness_metrics(
    tokens: Sequence[str],
    mattr_window: int = 100,
    mtld_threshold: float = 0.72,
    hdd_draws: int = 42,
) -> dict[str, float | None]:
    """
    Compute lexical diversity metrics via LexicalRichness with safe fallbacks.
    """
    if not tokens:
        return {
            "yules_k": 0.0,
            "mtld": 0.0,
            "mattr": 0.0,
            "hdd": 0.0,
        }

    try:
        from lexicalrichness import LexicalRichness  # type: ignore
    except Exception:
        # Fallback to internal implementations if dependency unavailable
        return {
            "yules_k": calculate_yules_k(tokens),
            "mtld": calculate_mtld(tokens, threshold=mtld_threshold),
            "mattr": calculate_mattr(tokens, window_size=mattr_window),
            "hdd": 0.0,
        }

    # LexicalRichness expects a string, not a list
    text_str = " ".join(tokens)
    lex = LexicalRichness(text_str)

    try:
        yules_k = float(lex.yulek())
    except Exception:
        yules_k = calculate_yules_k(tokens)

    try:
        mtld = float(lex.mtld(threshold=mtld_threshold))
    except Exception:
        mtld = calculate_mtld(tokens, threshold=mtld_threshold)

    try:
        mattr = float(lex.mattr(window_size=mattr_window))
    except Exception:
        mattr = calculate_mattr(tokens, window_size=mattr_window)

    try:
        hdd = float(lex.hdd(draws=hdd_draws))
    except Exception:
        hdd = 0.0

    return {
        "yules_k": yules_k,
        "mtld": mtld,
        "mattr": mattr,
        "hdd": hdd,
    }


def calculate_burstiness_index(tokens: Sequence[str], min_count: int = 5) -> float:
    """
    Calculate average Burstiness (B) index for words.

    B = (σ - μ) / (σ + μ) of inter-arrival times.
    Range: -1 (perfectly regular) to +1 (maximally bursty). 0 = Poisson/random.

    Args:
        tokens: List of tokens/words
        min_count: Minimum occurrences of word to be considered

    Returns:
        Average Burstiness Index across all qualifying words
    """
    positions: dict[str, list[int]] = {}
    for i, token in enumerate(tokens):
        if token not in positions:
            positions[token] = []
        positions[token].append(i)

    b_values = []

    for token, pos_list in positions.items():
        if len(pos_list) < min_count:
            continue

        # Calculate intervals between consecutive occurrences
        intervals = np.diff(pos_list)
        if len(intervals) < 2:
            continue

        mu = np.mean(intervals)
        # Use sample std (ddof=1) for unbiased estimate
        sigma = np.std(intervals, ddof=1)

        if mu + sigma == 0:
            continue

        # Burstiness: B = (sigma - mu) / (sigma + mu)
        b = (sigma - mu) / (sigma + mu)
        b_values.append(b)

    if not b_values:
        return 0.0

    return float(np.mean(b_values))


@dataclass
class BurstinessMetrics:
    """Comprehensive burstiness analysis results."""
    
    # Simple burstiness (baseline)
    burstiness_b: float  # (σ - μ) / (σ + μ)
    
    # Kleinberg's algorithm (pybursts)
    kleinberg_bursts: int | None  # Number of detected bursts
    kleinberg_burst_ratio: float | None  # Ratio of bursty periods
    
    # Bursty dynamics metrics
    burstiness_parameter: float | None  # BP from bursty_dynamics
    memory_coefficient: float | None  # MC from bursty_dynamics


def calculate_advanced_burstiness_metrics(
    tokens: Sequence[str],
    min_count: int = 5,
) -> BurstinessMetrics:
    """
    Calculate comprehensive burstiness metrics using pybursts and bursty_dynamics.
    
    Combines:
    - Simple burstiness (B) as baseline
    - Kleinberg's algorithm for burst detection
    - Burstiness parameter and memory coefficient from bursty_dynamics
    
    Args:
        tokens: List of tokens/words
        min_count: Minimum occurrences of word to be considered
        
    Returns:
        BurstinessMetrics with all computed metrics
    """
    # Baseline: simple burstiness
    burstiness_b = calculate_burstiness_index(tokens, min_count=min_count)
    
    # Initialize optional metrics
    kleinberg_bursts = None
    kleinberg_burst_ratio = None
    burstiness_parameter = None
    memory_coefficient = None
    
    if not tokens:
        return BurstinessMetrics(
            burstiness_b=0.0,
            kleinberg_bursts=None,
            kleinberg_burst_ratio=None,
            burstiness_parameter=None,
            memory_coefficient=None,
        )
    
    # Kleinberg's algorithm via pybursts
    # Note: pybursts has known issues with certain input formats
    # We use a fallback manual burst detection if it fails
    try:
        from pybursts.pybursts import kleinberg  # type: ignore
        
        token_counts = Counter(tokens)
        frequent_tokens = {token for token, count in token_counts.items() if count >= min_count}
        
        if frequent_tokens:
            most_frequent = max(frequent_tokens, key=lambda t: token_counts[t])
            positions = [i for i, token in enumerate(tokens) if token == most_frequent]
            
            if len(positions) >= min_count:
                try:
                    # pybursts expects offsets (timestamps), convert positions to float
                    offsets = np.array(positions, dtype=np.float64)
                    bursts = kleinberg(offsets, s=2, gamma=0.5)
                    
                    if bursts is not None and len(bursts) > 0:
                        # bursts is array of [level, start, end] tuples
                        if isinstance(bursts, np.ndarray) and bursts.ndim == 2:
                            burst_levels = bursts[:, 0]
                            kleinberg_bursts = int(np.sum(burst_levels > 0))
                            kleinberg_burst_ratio = float(kleinberg_bursts / len(burst_levels)) if len(burst_levels) > 0 else 0.0
                        else:
                            # Fallback for unexpected format
                            kleinberg_bursts = 0
                            kleinberg_burst_ratio = 0.0
                    else:
                        kleinberg_bursts = 0
                        kleinberg_burst_ratio = 0.0
                except (IndexError, ValueError, TypeError):
                    # pybursts has bugs with certain inputs - use manual fallback
                    # Manual burst detection: count clusters of occurrences
                    intervals = np.diff(positions)
                    if len(intervals) > 0:
                        median_interval = np.median(intervals)
                        # A "burst" is when interval is less than half the median
                        burst_threshold = median_interval * 0.5
                        burst_count = int(np.sum(intervals < burst_threshold))
                        kleinberg_bursts = burst_count
                        kleinberg_burst_ratio = float(burst_count / len(intervals)) if len(intervals) > 0 else 0.0
                    else:
                        kleinberg_bursts = 0
                        kleinberg_burst_ratio = 0.0
    except ImportError:
        # pybursts not available - use manual burst detection
        token_counts = Counter(tokens)
        frequent_tokens = {token for token, count in token_counts.items() if count >= min_count}
        if frequent_tokens:
            most_frequent = max(frequent_tokens, key=lambda t: token_counts[t])
            positions = [i for i, token in enumerate(tokens) if token == most_frequent]
            if len(positions) >= min_count:
                intervals = np.diff(positions)
                if len(intervals) > 0:
                    median_interval = np.median(intervals)
                    burst_threshold = median_interval * 0.5
                    burst_count = int(np.sum(intervals < burst_threshold))
                    kleinberg_bursts = burst_count
                    kleinberg_burst_ratio = float(burst_count / len(intervals))
    except Exception:
        # Other errors
        pass
    
    # Bursty dynamics metrics
    # Note: bursty_dynamics API expects DataFrame, so we'll compute manually
    # BP = (σ - μ) / (σ + μ) where σ is std of intervals, μ is mean
    # MC = correlation between consecutive intervals
    try:
        # Build inter-arrival times for frequent words
        token_positions: dict[str, list[int]] = {}
        for i, token in enumerate(tokens):
            if token not in token_positions:
                token_positions[token] = []
            token_positions[token].append(i)
        
        # Collect all inter-arrival intervals from frequent words
        all_intervals: list[float] = []
        for token, positions_list in token_positions.items():
            if len(positions_list) >= min_count:
                intervals = np.diff(positions_list).astype(float)
                if len(intervals) > 0:
                    all_intervals.extend(intervals.tolist())
        
        if len(all_intervals) >= 3:
            intervals_array = np.array(all_intervals)
            
            try:
                # Burstiness Parameter: BP = (σ - μ) / (σ + μ)
                mu = np.mean(intervals_array)
                sigma = np.std(intervals_array, ddof=1)
                
                if mu + sigma > 0:
                    bp_val = (sigma - mu) / (sigma + mu)
                    burstiness_parameter = float(bp_val) if np.isfinite(bp_val) else None
                else:
                    burstiness_parameter = None
                
                # Memory Coefficient: correlation between consecutive intervals
                if len(intervals_array) >= 2:
                    # Create pairs of consecutive intervals
                    intervals_1 = intervals_array[:-1]
                    intervals_2 = intervals_array[1:]
                    
                    if len(intervals_1) >= 2:
                        # Compute Pearson correlation
                        correlation_matrix = np.corrcoef(intervals_1, intervals_2)
                        mc_val = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0.0
                        memory_coefficient = float(mc_val) if np.isfinite(mc_val) else None
                    else:
                        memory_coefficient = None
                else:
                    memory_coefficient = None
            except Exception:
                # Fallback if computation fails
                pass
    except Exception:
        # Other errors
        pass
    
    return BurstinessMetrics(
        burstiness_b=burstiness_b,
        kleinberg_bursts=kleinberg_bursts,
        kleinberg_burst_ratio=kleinberg_burst_ratio,
        burstiness_parameter=burstiness_parameter,
        memory_coefficient=memory_coefficient,
    )


def calculate_mattr(
    tokens: Sequence[str],
    window_size: int = 500,
) -> float:
    """
    Calculate Moving Average Type-Token Ratio (MATTR).

    MATTR solves the problem of TTR sensitivity to text structure by using
    a sliding window approach. This is the most "smooth" and statistically
    sound metric for short texts.

    Formula: MATTR = (1 / (N - L + 1)) * sum(TTR(window_i))
    where TTR(window_i) = unique_tokens / total_tokens in window

    Args:
        tokens: List of tokens/words
        window_size: Size of sliding window (default 500, use 50-100 for short texts)

    Returns:
        MATTR value (0-1, where 1 = maximum diversity)
    """
    if not tokens:
        return 0.0

    n = len(tokens)
    if n < window_size:
        # For very short texts, use the entire text as one window
        unique = len(set(tokens))
        return float(unique / n) if n > 0 else 0.0

    # Optimized sliding window using Counter with incremental updates
    # This avoids O(N * L) complexity
    window_counter: Counter[str] = Counter()
    ttr_sum = 0.0
    num_windows = n - window_size + 1

    # Initialize first window
    for i in range(window_size):
        window_counter[tokens[i]] += 1

    # Calculate TTR for first window
    unique_count = len(window_counter)
    ttr_sum += unique_count / window_size

    # Slide window: remove leftmost, add rightmost
    for i in range(1, num_windows):
        # Remove leftmost token
        left_token = tokens[i - 1]
        window_counter[left_token] -= 1
        if window_counter[left_token] == 0:
            del window_counter[left_token]

        # Add rightmost token
        right_token = tokens[i + window_size - 1]
        window_counter[right_token] += 1

        # Calculate TTR for current window
        unique_count = len(window_counter)
        ttr_sum += unique_count / window_size

    return float(ttr_sum / num_windows)


@dataclass
class DispersionResult:
    """Result of dispersion analysis for a word."""

    dp: float  # Deviation of Proportions (0 = perfect dispersion, ~1 = concentrated)
    dp_norm: float  # Normalized DP (0-1, where 1 = perfect dispersion)
    word: str
    total_frequency: int


def calculate_gries_dp(
    tokens: Sequence[str],
    word: str,
    n_segments: int = 10,
) -> DispersionResult:
    """
    Calculate Gries' Deviation of Proportions (DP) for a word.

    DP measures how evenly a word is distributed across text segments.
    It is insensitive to corpus size, unlike earlier metrics like Juilland's D.

    Formula: DP = (1/2) * sum(|v_i/F - s_i/S|)
    where:
    - v_i = frequency of word in segment i
    - F = total frequency of word
    - s_i = size of segment i in words
    - S = total text size

    Normalized: DP_norm = 1 - DP / (1 - min(s_i/S))

    Args:
        tokens: List of tokens/words
        word: Word to analyze
        n_segments: Number of segments to divide text into

    Returns:
        DispersionResult with DP and normalized DP
    """
    if not tokens:
        return DispersionResult(dp=0.0, dp_norm=1.0, word=word, total_frequency=0)

    n = len(tokens)
    if n_segments <= 0:
        n_segments = 1

    # Limit segments to number of tokens (can't have empty segments)
    n_segments = min(n_segments, n)

    # Calculate segment sizes
    segment_size = n // n_segments
    remainder = n % n_segments

    # Build segments with sizes (some may be 1 token larger due to remainder)
    segments: list[list[str]] = []
    start = 0
    for i in range(n_segments):
        size = segment_size + (1 if i < remainder else 0)
        end = start + size
        segments.append(tokens[start:end])
        start = end

    # Count word frequency in each segment
    segment_freqs: list[int] = []
    total_freq = 0

    for segment in segments:
        freq = segment.count(word)
        segment_freqs.append(freq)
        total_freq += freq

    if total_freq == 0:
        # Word not found
        return DispersionResult(dp=0.0, dp_norm=1.0, word=word, total_frequency=0)

    # Calculate segment sizes in tokens
    segment_sizes = [len(seg) for seg in segments]
    total_size = sum(segment_sizes)

    if total_size == 0:
        return DispersionResult(dp=0.0, dp_norm=1.0, word=word, total_frequency=total_freq)

    # Calculate DP
    dp_sum = 0.0
    min_segment_prop = min(s / total_size for s in segment_sizes)

    for v_i, s_i in zip(segment_freqs, segment_sizes):
        observed_prop = v_i / total_freq if total_freq > 0 else 0.0
        expected_prop = s_i / total_size
        dp_sum += abs(observed_prop - expected_prop)

    dp = 0.5 * dp_sum

    # Normalize: DP_norm = 1 - DP / (1 - min(s_i/S))
    # This gives us a metric where 1 = perfect dispersion
    if min_segment_prop >= 1.0:
        dp_norm = 1.0 - dp
    else:
        dp_norm = 1.0 - (dp / (1.0 - min_segment_prop))

    # Clamp to [0, 1]
    dp_norm = max(0.0, min(1.0, dp_norm))

    return DispersionResult(
        dp=float(dp),
        dp_norm=float(dp_norm),
        word=word,
        total_frequency=total_freq,
    )


def calculate_average_gries_dp(
    tokens: Sequence[str],
    n_segments: int = 10,
    min_word_freq: int = 5,
) -> float:
    """
    Calculate average Gries' DP across all words in text.

    This provides a global measure of word dispersion for the entire text.

    Args:
        tokens: List of tokens/words
        n_segments: Number of segments to divide text into
        min_word_freq: Minimum frequency for word to be included in average

    Returns:
        Average normalized DP (0-1, where 1 = perfect dispersion)
    """
    if not tokens:
        return 0.0

    # Count word frequencies
    word_counts = Counter(tokens)

    # Filter words by minimum frequency
    words_to_analyze = [word for word, count in word_counts.items() if count >= min_word_freq]

    if not words_to_analyze:
        return 0.0

    # Calculate DP for each word
    dp_norm_values = []
    for word in words_to_analyze:
        result = calculate_gries_dp(tokens, word, n_segments)
        dp_norm_values.append(result.dp_norm)

    if not dp_norm_values:
        return 0.0

    return float(np.mean(dp_norm_values))


@dataclass
class HurstResult:
    """Result of Hurst exponent calculation via DFA."""

    hurst_exponent: float  # H (0 < H < 1)
    alpha: float  # Same as H (DFA terminology)
    r_squared: float  # Goodness of fit for log-log regression
    interpretation: str  # "persistent", "random", "anti-persistent"


def calculate_hurst_exponent_dfa(
    time_series: Sequence[float] | NDArray[np.floating],
    min_scale: int = 4,
    max_scale: int | None = None,
    polynomial_order: int = 1,
) -> HurstResult:
    """
    Calculate Hurst exponent using Detrended Fluctuation Analysis (DFA).

    The Hurst exponent H characterizes the degree of persistence (memory) in a time series:
    - H = 0.5: Random process (white noise)
    - H > 0.5: Persistent process (long memory) - fractal structure
    - H < 0.5: Anti-persistent (mean-reverting)

    Algorithm (DFA):
    1. Integrate: Y(k) = sum(x_t - mean(x)) from t=1 to k
    2. Segment: Divide Y(k) into non-overlapping windows of size s
    3. Detrend: Fit polynomial in each window and subtract
    4. Fluctuation: F(s) = sqrt(mean(Y_detrend^2))
    5. Scaling: Repeat for different scales s
    6. Result: H = slope of log(F(s)) vs log(s)

    For reliable DFA estimates, we need at least 2-3 orders of magnitude in scale.
    The algorithm uses geometric progression to sample scales efficiently.

    Args:
        time_series: Input time series (e.g., word lengths)
        min_scale: Minimum window size (default 4)
        max_scale: Maximum window size (default: len/4, no artificial cap)
        polynomial_order: Order of detrending polynomial (1 = linear, DFA1)

    Returns:
        HurstResult with H, alpha, R², and interpretation
    """
    series = np.asarray(time_series, dtype=np.float64)
    n = len(series)

    if n < min_scale * 2:
        return HurstResult(
            hurst_exponent=0.5,
            alpha=0.5,
            r_squared=0.0,
            interpretation="insufficient_data",
        )

    # Step 1: Integrate (cumulative sum with mean subtracted)
    mean_val = np.mean(series)
    integrated = np.cumsum(series - mean_val)

    # Step 2-5: Calculate F(s) for different scales
    # For reliable DFA, max_scale should be n/4 to ensure enough windows
    if max_scale is None:
        max_scale = n // 4

    scales = []
    fluctuations = []

    # Use geometric progression with factor ~1.2 for smoother sampling
    # Target ~20-25 scale points for good regression quality
    max_scales = 25
    s = min_scale
    while s <= max_scale and s <= n // 2 and len(scales) < max_scales:
        scales.append(s)
        # Use factor 1.2 for smoother progression (was 1.3)
        s = max(s + 1, int(s * 1.2))

    if len(scales) < 4:
        # Fallback: use linear progression
        step = max(1, (max_scale - min_scale) // max_scales)
        scales = list(range(min_scale, min(max_scale, n // 2) + 1, max(1, step)))[:max_scales]

    for scale in scales:
        if scale > n:
            continue

        # Divide into non-overlapping windows
        n_windows = n // scale
        if n_windows < 2:
            continue

        window_fluctuations = []

        for i in range(n_windows):
            start = i * scale
            end = start + scale
            window_data = integrated[start:end]

            if len(window_data) < polynomial_order + 1:
                continue

            # Detrend: fit polynomial and subtract
            x_window = np.arange(len(window_data))
            coeffs = np.polyfit(x_window, window_data, polynomial_order)
            trend = np.polyval(coeffs, x_window)
            detrended = window_data - trend

            # Calculate fluctuation for this window
            fluctuation = np.sqrt(np.mean(detrended**2))
            window_fluctuations.append(fluctuation)

        if window_fluctuations:
            # Average fluctuation across all windows
            f_s = np.mean(window_fluctuations)
            fluctuations.append(f_s)

    if len(scales) < 3 or len(fluctuations) < 3:
        return HurstResult(
            hurst_exponent=0.5,
            alpha=0.5,
            r_squared=0.0,
            interpretation="insufficient_data",
        )

    # Step 6: Linear regression on log-log plot
    log_scales = np.log(np.array(scales[: len(fluctuations)]))
    log_fluctuations = np.log(np.array(fluctuations))

    # Remove any invalid values
    valid = np.isfinite(log_scales) & np.isfinite(log_fluctuations)
    if np.sum(valid) < 3:
        return HurstResult(
            hurst_exponent=0.5,
            alpha=0.5,
            r_squared=0.0,
            interpretation="insufficient_data",
        )

    log_scales = log_scales[valid]
    log_fluctuations = log_fluctuations[valid]

    slope, intercept, r_value, _, _ = stats.linregress(log_scales, log_fluctuations)
    hurst = float(slope)

    # Interpret result
    if hurst > 0.6:
        interpretation = "persistent"
    elif hurst > 0.4:
        interpretation = "random"
    else:
        interpretation = "anti-persistent"

    return HurstResult(
        hurst_exponent=float(hurst),
        alpha=float(hurst),
        r_squared=float(r_value**2),
        interpretation=interpretation,
    )


def calculate_hurst_from_word_lengths(tokens: Sequence[str]) -> HurstResult:
    """
    Calculate Hurst exponent from word length time series.

    Converts tokens to a time series of word lengths, then applies DFA.

    Args:
        tokens: List of tokens/words

    Returns:
        HurstResult with H exponent
    """
    if not tokens:
        return HurstResult(
            hurst_exponent=0.5,
            alpha=0.5,
            r_squared=0.0,
            interpretation="insufficient_data",
        )

    word_lengths = [len(token) for token in tokens]
    return calculate_hurst_exponent_dfa(word_lengths)


@dataclass
class ZipfMandelbrotAnalysis:
    """Result of Zipf-Mandelbrot law analysis."""

    alpha: float  # Zipf exponent
    beta: float  # Mandelbrot parameter (curvature correction)
    c: float  # Normalization constant
    r_squared: float  # Goodness of fit
    r_squared_improvement: float  # Improvement over simple Zipf


def calculate_zipf_mandelbrot(
    counts: Sequence[int] | NDArray[np.integer],
    adaptive_sampling: bool = True,
) -> ZipfMandelbrotAnalysis:
    """
    Analyze distribution using Zipf-Mandelbrot law.

    Zipf-Mandelbrot: f(r) = C / (r + beta)^alpha

    This is more accurate than simple Zipf's law, especially for the
    most frequent words. The beta parameter corrects curvature at the
    beginning of the distribution.

    Args:
        counts: Array of counts (will be sorted by frequency)
        adaptive_sampling: If True, uses intelligent sampling that preserves
            distribution shape (logarithmic sampling for tail, dense for head)

    Returns:
        ZipfMandelbrotAnalysis with alpha, beta, C, and fit quality
    """
    counts_arr = np.asarray(counts, dtype=np.float64)

    # Sort by frequency (descending)
    sorted_counts = np.sort(counts_arr)[::-1]

    # Remove zeros
    sorted_counts = sorted_counts[sorted_counts > 0]

    if len(sorted_counts) < 5:
        return ZipfMandelbrotAnalysis(
            alpha=0.0,
            beta=0.0,
            c=0.0,
            r_squared=0.0,
            r_squared_improvement=0.0,
        )

    # Ranks (1, 2, 3, ...)
    ranks = np.arange(1, len(sorted_counts) + 1, dtype=np.float64)
    n_total = len(sorted_counts)

    # Adaptive sampling: more points in the head (important for Mandelbrot correction),
    # logarithmic spacing in the tail (where Zipf behavior dominates)
    if adaptive_sampling and n_total > 300:
        # Strategy: 
        # - Keep all points in top 50 (critical for beta estimation)
        # - Logarithmic sampling for the rest
        n_head = min(50, n_total // 3)
        n_tail_samples = 250  # Total samples from tail
        
        head_indices = np.arange(n_head)
        
        # Logarithmic sampling for tail: more points near head, fewer at far tail
        tail_start = n_head
        tail_end = n_total - 1
        
        if tail_end > tail_start:
            # Log-spaced indices (excluding already-included head)
            tail_indices = np.unique(
                np.geomspace(tail_start, tail_end, n_tail_samples).astype(int)
            )
            # Ensure we include the very last point
            if tail_indices[-1] != tail_end:
                tail_indices = np.append(tail_indices, tail_end)
        else:
            tail_indices = np.array([], dtype=int)
        
        indices = np.concatenate([head_indices, tail_indices])
        sorted_counts = sorted_counts[indices]
        ranks = ranks[indices]

    # Simple Zipf for comparison
    log_ranks = np.log(ranks)
    log_counts = np.log(sorted_counts)
    zipf_slope, zipf_intercept, zipf_r, _, _ = stats.linregress(log_ranks, log_counts)
    zipf_r_squared = zipf_r**2

    # Zipf-Mandelbrot: f(r) = C / (r + beta)^alpha
    # Taking logs: log(f) = log(C) - alpha * log(r + beta)
    # This requires nonlinear fitting

    def zipf_mandelbrot_func(r, alpha, beta, log_c):
        """Zipf-Mandelbrot function for curve fitting."""
        return log_c - alpha * np.log(r + beta)

    # Initial guess: use simple Zipf parameters
    initial_alpha = -zipf_slope
    initial_beta = 1.0  # Common starting value
    initial_log_c = zipf_intercept

    try:
        # Weight points: higher weight for head (more important for beta)
        # and tail endpoints (important for alpha)
        weights = np.ones(len(ranks))
        weights[:min(20, len(weights))] = 2.0  # Head gets 2x weight
        weights[-min(10, len(weights)):] = 1.5  # Far tail gets 1.5x weight
        
        # Fit using curve_fit with adaptive iterations
        max_iterations = min(2000, 500 + len(ranks) * 5)
        
        popt, _ = curve_fit(
            zipf_mandelbrot_func,
            ranks,
            log_counts,
            p0=[initial_alpha, initial_beta, initial_log_c],
            bounds=([0.1, 0.0, -np.inf], [5.0, 100.0, np.inf]),
            sigma=1.0 / weights,  # Inverse weights for sigma
            maxfev=max_iterations,
        )

        alpha_fit, beta_fit, log_c_fit = popt
        c_fit = np.exp(log_c_fit)

        # Calculate R² (weighted)
        predicted_log = zipf_mandelbrot_func(ranks, alpha_fit, beta_fit, log_c_fit)
        residuals = log_counts - predicted_log
        ss_res = np.sum(weights * residuals ** 2)
        ss_tot = np.sum(weights * (log_counts - np.average(log_counts, weights=weights)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        r_squared_improvement = max(0.0, r_squared - zipf_r_squared)

    except (RuntimeError, ValueError):
        # Fallback to simple Zipf if fitting fails
        alpha_fit = initial_alpha
        beta_fit = 0.0
        c_fit = np.exp(initial_log_c)
        r_squared = zipf_r_squared
        r_squared_improvement = 0.0

    return ZipfMandelbrotAnalysis(
        alpha=float(alpha_fit),
        beta=float(beta_fit),
        c=float(c_fit),
        r_squared=float(r_squared),
        r_squared_improvement=float(r_squared_improvement),
    )


@dataclass
class PierrehumbertBetaResult:
    """Result of Pierrehumbert's beta (Weibull distribution) analysis."""

    beta: float  # Shape parameter (β)
    scale: float  # Scale parameter (λ)
    r_squared: float  # Goodness of fit
    interpretation: str  # "content_word", "function_word", "poisson"


def calculate_pierrehumbert_beta(
    tokens: Sequence[str],
    word: str,
    min_occurrences: int = 5,
) -> PierrehumbertBetaResult | None:
    """
    Calculate Pierrehumbert's beta parameter for a word using Weibull distribution.

    The distribution of intervals between word occurrences follows:
    P(t) ~ e^(-(t/λ)^β)

    Where:
    - β = 1: Exponential (Poisson process, no memory)
    - β < 1: Fat tail (bursty, content words)
    - β > 1: Regular (function words)

    Args:
        tokens: List of tokens/words
        word: Word to analyze
        min_occurrences: Minimum occurrences required

    Returns:
        PierrehumbertBetaResult or None if insufficient data
    """
    if not tokens:
        return None

    # Find positions of word
    positions = [i for i, token in enumerate(tokens) if token == word]

    if len(positions) < min_occurrences:
        return None

    # Calculate intervals between occurrences
    intervals = np.diff(positions)
    intervals = intervals[intervals > 0]  # Remove zeros

    if len(intervals) < 3:
        return None

    # Fit Weibull distribution using MLE
    # scipy.stats.weibull_min uses: f(x) = (c/λ) * (x/λ)^(c-1) * exp(-(x/λ)^c)
    # Where c is shape parameter (our β) and λ is scale
    try:
        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(intervals, floc=0)

        # Calculate R² (goodness of fit)
        # Compare empirical CDF with fitted CDF
        sorted_intervals = np.sort(intervals)
        empirical_cdf = np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals)
        fitted_cdf = stats.weibull_min.cdf(sorted_intervals, shape, loc=0, scale=scale)

        # R² for CDF fit
        ss_res = np.sum((empirical_cdf - fitted_cdf) ** 2)
        ss_tot = np.sum((empirical_cdf - np.mean(empirical_cdf)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Interpret beta
        if shape < 0.7:
            interpretation = "content_word"
        elif shape < 1.3:
            interpretation = "poisson"
        else:
            interpretation = "function_word"

        return PierrehumbertBetaResult(
            beta=float(shape),
            scale=float(scale),
            r_squared=float(r_squared),
            interpretation=interpretation,
        )

    except (RuntimeError, ValueError, TypeError):
        return None


@dataclass
class ReadabilityResult:
    """Result of readability analysis."""

    flesch_reading_ease: float  # Flesch Reading Ease (0-100+, higher = easier)
    gunning_fog_index: float  # Gunning Fog Index (grade level)
    avg_sentence_length: float  # Average words per sentence
    avg_syllables_per_word: float  # Average syllables per word
    complex_word_percentage: float  # Percentage of words with 3+ syllables
    n_sentences: int  # Number of sentences
    n_words: int  # Number of words
    n_syllables: int  # Total syllables


# Russian vowels for syllable counting
_RUSSIAN_VOWELS = frozenset("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


def count_syllables_russian(word: str) -> int:
    """
    Count syllables in a Russian word.

    In Russian, the number of syllables equals the number of vowels.

    Args:
        word: Russian word

    Returns:
        Number of syllables (at least 1 for non-empty words)
    """
    if not word:
        return 0

    count = sum(1 for char in word if char in _RUSSIAN_VOWELS)
    # Every word has at least 1 syllable if it has letters
    return max(1, count) if any(c.isalpha() for c in word) else 0


@dataclass
class SyllableEntropyResult:
    """Result of syllable-based entropy analysis."""
    
    # Syllable count distribution entropy
    syllable_count_entropy: float  # Entropy of word-length distribution (in syllables)
    syllable_count_conditional: float  # H(next syllable count | previous)
    
    # Statistics
    total_syllables: int
    total_words: int
    avg_syllables_per_word: float
    syllable_count_distribution: dict[int, int]  # count -> frequency
    
    # Syllable pattern entropy (for Russian: vowel patterns)
    vowel_pattern_entropy: float  # Entropy of vowel patterns within words
    unique_patterns: int


def calculate_syllable_entropy(tokens: Sequence[str]) -> SyllableEntropyResult | None:
    """
    Calculate entropy based on syllable structure of Russian text.
    
    For Russian, syllables are determined by vowels. This function analyzes:
    1. Distribution of syllable counts per word (1-syllable, 2-syllable, etc.)
    2. Conditional entropy: how predictable is the syllable count given previous word
    3. Vowel patterns within words (structural fingerprint)
    
    This metric is particularly informative for Russian because:
    - Russian has rich morphology with prefixes/suffixes affecting syllable count
    - Different authors/genres have characteristic syllable distributions
    - Poetry vs prose has very different syllable patterns
    
    Args:
        tokens: List of words/tokens
        
    Returns:
        SyllableEntropyResult or None if insufficient data
    """
    if not tokens or len(tokens) < 10:
        return None
    
    # Count syllables per word
    syllable_counts = [count_syllables_russian(word) for word in tokens]
    
    # Filter out words with 0 syllables (numbers, punctuation that slipped through)
    syllable_counts = [s for s in syllable_counts if s > 0]
    
    if len(syllable_counts) < 10:
        return None
    
    total_syllables = sum(syllable_counts)
    total_words = len(syllable_counts)
    avg_syllables = total_syllables / total_words if total_words > 0 else 0.0
    
    # Distribution of syllable counts
    syllable_count_dist = Counter(syllable_counts)
    
    # Entropy of syllable count distribution
    counts_array = np.array(list(syllable_count_dist.values()), dtype=np.int64)
    syllable_count_entropy = _entropy_from_counts(counts_array)
    
    # Conditional entropy: H(syllable_count[i] | syllable_count[i-1])
    # Build bigrams of syllable counts
    if len(syllable_counts) >= 2:
        syllable_bigrams = Counter(
            (syllable_counts[i], syllable_counts[i + 1])
            for i in range(len(syllable_counts) - 1)
        )
        context_counts = Counter(syllable_counts[:-1])
        
        bigram_array = np.array(list(syllable_bigrams.values()), dtype=np.int64)
        context_array = np.array(list(context_counts.values()), dtype=np.int64)
        
        h_bigram = _entropy_from_counts(bigram_array)
        h_context = _entropy_from_counts(context_array)
        syllable_count_conditional = max(0.0, h_bigram - h_context)
    else:
        syllable_count_conditional = 0.0
    
    # Vowel pattern entropy: extract vowel patterns from words
    # Pattern = sequence of vowels in word, e.g., "привет" -> "ие"
    vowel_patterns: list[str] = []
    for word in tokens:
        pattern = "".join(c.lower() for c in word if c.lower() in "аеёиоуыэюя")
        if pattern:  # Only words with vowels
            vowel_patterns.append(pattern)
    
    if vowel_patterns:
        pattern_counts = Counter(vowel_patterns)
        pattern_array = np.array(list(pattern_counts.values()), dtype=np.int64)
        vowel_pattern_entropy = _entropy_from_counts(pattern_array)
        unique_patterns = len(pattern_counts)
    else:
        vowel_pattern_entropy = 0.0
        unique_patterns = 0
    
    return SyllableEntropyResult(
        syllable_count_entropy=float(syllable_count_entropy),
        syllable_count_conditional=float(syllable_count_conditional),
        total_syllables=total_syllables,
        total_words=total_words,
        avg_syllables_per_word=float(avg_syllables),
        syllable_count_distribution=dict(syllable_count_dist),
        vowel_pattern_entropy=float(vowel_pattern_entropy),
        unique_patterns=unique_patterns,
    )


def count_sentences(text: str) -> int:
    """
    Count sentences in text.

    Uses heuristics to handle:
    - Common Russian abbreviations (т.е., т.д., т.п., г., др., пр.)
    - Initials (А.С. Пушкин)
    - Multiple punctuation marks (..., ?!, etc.)

    Args:
        text: Input text

    Returns:
        Number of sentences (at least 1 for non-empty text)
    """
    if not text.strip():
        return 0

    import re

    # Normalize text: replace common abbreviations with placeholders
    # to prevent false sentence breaks
    abbreviations = [
        (r'\bт\.е\.', 'т·е·'),
        (r'\bт\.д\.', 'т·д·'),
        (r'\bт\.п\.', 'т·п·'),
        (r'\bт\.к\.', 'т·к·'),
        (r'\bи т\.д\.', 'и т·д·'),
        (r'\bи т\.п\.', 'и т·п·'),
        (r'\bи др\.', 'и др·'),
        (r'\bи пр\.', 'и пр·'),
        (r'\bг\.', 'г·'),  # год
        (r'\bгг\.', 'гг·'),  # годы
        (r'\bв\.', 'в·'),  # век
        (r'\bвв\.', 'вв·'),  # века
        (r'\bстр\.', 'стр·'),
        (r'\bс\.', 'с·'),  # страница
        (r'\bсм\.', 'см·'),
        (r'\bср\.', 'ср·'),
        (r'\bпр\.', 'пр·'),
        (r'\bдр\.', 'др·'),
        (r'\bобл\.', 'обл·'),
    ]

    processed = text
    for pattern, replacement in abbreviations:
        processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

    # Handle initials: single uppercase letter followed by period
    # e.g., "А.С. Пушкин" -> "А·С· Пушкин"
    processed = re.sub(r'\b([А-ЯЁA-Z])\.', r'\1·', processed)

    # Now split by sentence-ending punctuation
    # Match one or more of .!? but not single dots that might be decimal points
    sentence_endings = re.split(r'[.!?]+(?:\s|$)', processed)

    # Filter out empty strings
    sentences = [s.strip() for s in sentence_endings if s.strip()]

    return max(1, len(sentences))


def calculate_readability(
    tokens: Sequence[str],
    text: str,
) -> ReadabilityResult:
    """
    Calculate readability indices for Russian text.

    Calculates:
    - Flesch Reading Ease (heuristically adapted for Russian)
    - Gunning Fog Index

    **WARNING**: The Flesch formula was originally calibrated for English.
    The coefficients used here (1.3, 60.1) are heuristic adaptations and
    have NOT been empirically validated on Russian text corpora.
    Treat these scores as rough estimates, not authoritative measures.

    Flesch Reading Ease interpretation (approximate for Russian):
    - 90-100: Very easy (for elementary school children)
    - 80-90: Easy (for 5-8 graders)
    - 70-80: Fairly easy (for 9-11 graders)
    - 60-70: Medium difficulty (for college freshmen)
    - 50-60: Fairly difficult (for college students)
    - 30-50: Difficult (for specialists with higher education)
    - 0-30: Very difficult (for experts only)

    Gunning Fog Index interpretation:
    - <6: Easy, suitable for wide audience
    - 7-12: Easy to read, understandable for most adults
    - 13-17: Medium difficulty, requires some effort
    - 18-24: High difficulty, for educated readers or specialists
    - >24: Very difficult, hard even for experts

    Args:
        tokens: List of words/tokens
        text: Original text (for sentence counting)

    Returns:
        ReadabilityResult with all metrics
    """
    n_words = len(tokens)
    if n_words == 0:
        return ReadabilityResult(
            flesch_reading_ease=0.0,
            gunning_fog_index=0.0,
            avg_sentence_length=0.0,
            avg_syllables_per_word=0.0,
            complex_word_percentage=0.0,
            n_sentences=0,
            n_words=0,
            n_syllables=0,
        )

    # Count sentences
    n_sentences = count_sentences(text)

    # Count syllables
    syllables_per_word = [count_syllables_russian(word) for word in tokens]
    n_syllables = sum(syllables_per_word)

    # Average sentence length (words per sentence)
    asl = n_words / n_sentences if n_sentences > 0 else n_words

    # Average syllables per word
    asw = n_syllables / n_words if n_words > 0 else 0.0

    # Complex words (3+ syllables)
    complex_words = sum(1 for s in syllables_per_word if s >= 3)
    complex_word_pct = (complex_words / n_words * 100) if n_words > 0 else 0.0

    # Flesch Reading Ease (adapted for Russian)
    # Original formula: 206.835 - 1.015 × ASL - 84.6 × ASW
    # Russian adaptation uses slightly different coefficients
    flesch = 206.835 - 1.3 * asl - 60.1 * asw

    # Gunning Fog Index
    # GFI = 0.4 × (ASL + PHW)
    # where PHW = percentage of complex words (3+ syllables)
    gunning_fog = 0.4 * (asl + complex_word_pct)

    return ReadabilityResult(
        flesch_reading_ease=float(flesch),
        gunning_fog_index=float(gunning_fog),
        avg_sentence_length=float(asl),
        avg_syllables_per_word=float(asw),
        complex_word_percentage=float(complex_word_pct),
        n_sentences=n_sentences,
        n_words=n_words,
        n_syllables=n_syllables,
    )


def calculate_average_pierrehumbert_beta(
    tokens: Sequence[str],
    min_word_freq: int = 10,
    max_words: int = 30,
) -> float | None:
    """
    Calculate average Pierrehumbert's beta across all words in text.

    Provides a global measure of burstiness for the entire text.

    Args:
        tokens: List of tokens/words
        min_word_freq: Minimum frequency for word to be included
        max_words: Maximum number of words to analyze (for performance)

    Returns:
        Average beta value or None if insufficient data
    """
    if not tokens:
        return None

    # Count word frequencies
    word_counts = Counter(tokens)

    # Filter words by minimum frequency and take top N most frequent
    words_to_analyze = [
        word for word, count in word_counts.items() if count >= min_word_freq
    ]
    
    # Сортируем по частоте и берем топ-N для ускорения
    words_to_analyze = sorted(
        words_to_analyze,
        key=lambda w: word_counts[w],
        reverse=True
    )[:max_words]

    if not words_to_analyze:
        return None

    # Calculate beta for each word
    beta_values = []
    for word in words_to_analyze:
        result = calculate_pierrehumbert_beta(tokens, word, min_occurrences=min_word_freq)
        if result is not None:
            beta_values.append(result.beta)

    if not beta_values:
        return None

    return float(np.mean(beta_values))


# ============================================================================
# N-gram Distribution Analysis (for 9.5/10 accuracy)
# ============================================================================


@dataclass
class NgramStats:
    """Statistics for a single n-gram."""
    
    ngram: str  # The n-gram as string (e.g., "пр", "при")
    count: int  # Absolute frequency
    frequency: float  # Relative frequency (probability)
    rank: int  # Rank by frequency (1 = most frequent)


@dataclass
class NgramDistributionResult:
    """Complete n-gram distribution analysis result."""
    
    n: int  # N-gram order (2 for bigrams, 3 for trigrams)
    total_ngrams: int  # Total number of n-grams in text
    unique_ngrams: int  # Number of unique n-grams
    
    # Top n-grams by frequency
    top_ngrams: list[NgramStats]
    
    # Entropy metrics
    entropy: float  # Shannon entropy of n-gram distribution
    conditional_entropy: float  # H(Xn | X1...Xn-1)
    
    # Zipf analysis for n-grams
    zipf_alpha: float  # Zipf exponent
    zipf_r_squared: float  # Goodness of fit
    
    # Distribution characteristics
    hapax_legomena: int  # N-grams appearing exactly once
    hapax_ratio: float  # Ratio of hapax to total unique
    coverage_top_10: float  # What fraction of text is covered by top 10 n-grams
    coverage_top_50: float  # What fraction of text is covered by top 50 n-grams


def calculate_ngram_distribution(
    tokens: Sequence[str],
    n: int = 2,
    top_k: int = 50,
) -> NgramDistributionResult | None:
    """
    Calculate comprehensive n-gram distribution analysis.
    
    Provides not just entropy, but full distribution statistics including:
    - Top-K most frequent n-grams with counts and frequencies
    - Zipf analysis for n-gram distribution
    - Hapax legomena (n-grams appearing once)
    - Coverage statistics
    
    Args:
        tokens: Sequence of tokens (letters or words)
        n: N-gram order (2 for bigrams, 3 for trigrams)
        top_k: Number of top n-grams to return
        
    Returns:
        NgramDistributionResult or None if insufficient data
    """
    if len(tokens) < n:
        return None
    
    # Generate n-grams
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    
    if not ngrams:
        return None
    
    # Count n-grams
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    unique_ngrams = len(ngram_counts)
    
    # Sort by frequency (descending)
    sorted_ngrams = ngram_counts.most_common()
    
    # Build top-K stats
    top_ngrams: list[NgramStats] = []
    for rank, (ngram_tuple, count) in enumerate(sorted_ngrams[:top_k], 1):
        ngram_str = "".join(ngram_tuple) if all(isinstance(t, str) and len(t) == 1 for t in ngram_tuple) else " ".join(ngram_tuple)
        freq = count / total_ngrams
        top_ngrams.append(NgramStats(
            ngram=ngram_str,
            count=count,
            frequency=freq,
            rank=rank,
        ))
    
    # Calculate entropy of n-gram distribution using stable xlogy
    counts_array = np.array(list(ngram_counts.values()), dtype=np.int64)
    entropy = _entropy_from_counts(counts_array)
    
    # Calculate conditional entropy H(Xn | X1...Xn-1)
    # H(Xn|X1...Xn-1) = H(X1...Xn) - H(X1...Xn-1)
    conditional_entropy = calculate_ngram_entropy(tokens, n)
    
    # Zipf analysis for n-grams
    sorted_counts = np.sort(counts_array)[::-1]
    sorted_counts = sorted_counts[sorted_counts > 0]
    
    zipf_alpha = 0.0
    zipf_r_squared = 0.0
    
    if len(sorted_counts) >= 3:
        ranks = np.arange(1, len(sorted_counts) + 1, dtype=np.float64)
        log_ranks = np.log(ranks)
        log_counts = np.log(sorted_counts)
        
        slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_counts)
        zipf_alpha = float(-slope)
        zipf_r_squared = float(r_value ** 2)
    
    # Hapax legomena (n-grams appearing exactly once)
    hapax_count = sum(1 for count in ngram_counts.values() if count == 1)
    hapax_ratio = hapax_count / unique_ngrams if unique_ngrams > 0 else 0.0
    
    # Coverage statistics
    cumsum = np.cumsum(sorted_counts)
    coverage_top_10 = float(cumsum[min(9, len(cumsum) - 1)] / total_ngrams) if len(cumsum) > 0 else 0.0
    coverage_top_50 = float(cumsum[min(49, len(cumsum) - 1)] / total_ngrams) if len(cumsum) > 0 else 0.0
    
    return NgramDistributionResult(
        n=n,
        total_ngrams=total_ngrams,
        unique_ngrams=unique_ngrams,
        top_ngrams=top_ngrams,
        entropy=entropy,
        conditional_entropy=conditional_entropy,
        zipf_alpha=zipf_alpha,
        zipf_r_squared=zipf_r_squared,
        hapax_legomena=hapax_count,
        hapax_ratio=float(hapax_ratio),
        coverage_top_10=coverage_top_10,
        coverage_top_50=coverage_top_50,
    )


@dataclass
class NgramComparisonResult:
    """Result of comparing n-gram distributions between two texts."""
    
    n: int  # N-gram order
    
    # Shared vs unique n-grams
    shared_ngrams: int  # N-grams present in both texts
    unique_to_text1: int  # N-grams only in text 1
    unique_to_text2: int  # N-grams only in text 2
    jaccard_similarity: float  # |intersection| / |union|
    
    # Distribution divergence
    js_divergence: float  # Jensen-Shannon divergence
    cosine_similarity: float  # Cosine similarity of frequency vectors
    
    # Top differing n-grams
    top_diff_text1: list[NgramStats]  # N-grams much more frequent in text 1
    top_diff_text2: list[NgramStats]  # N-grams much more frequent in text 2


def compare_ngram_distributions(
    tokens1: Sequence[str],
    tokens2: Sequence[str],
    n: int = 2,
    top_k_diff: int = 10,
) -> NgramComparisonResult | None:
    """
    Compare n-gram distributions between two texts.
    
    Provides detailed comparison including:
    - Shared vs unique n-grams
    - Distribution similarity metrics
    - Most distinguishing n-grams for each text
    
    Args:
        tokens1: Tokens from first text
        tokens2: Tokens from second text
        n: N-gram order
        top_k_diff: Number of top differentiating n-grams to return
        
    Returns:
        NgramComparisonResult or None if insufficient data
    """
    if len(tokens1) < n or len(tokens2) < n:
        return None
    
    # Generate n-grams
    ngrams1 = [tuple(tokens1[i:i + n]) for i in range(len(tokens1) - n + 1)]
    ngrams2 = [tuple(tokens2[i:i + n]) for i in range(len(tokens2) - n + 1)]
    
    if not ngrams1 or not ngrams2:
        return None
    
    # Count n-grams
    counts1 = Counter(ngrams1)
    counts2 = Counter(ngrams2)
    
    total1 = len(ngrams1)
    total2 = len(ngrams2)
    
    # Set operations
    set1 = set(counts1.keys())
    set2 = set(counts2.keys())
    
    shared = set1 & set2
    unique1 = set1 - set2
    unique2 = set2 - set1
    union = set1 | set2
    
    jaccard = len(shared) / len(union) if union else 0.0
    
    # Build frequency vectors for shared vocabulary + unique
    all_ngrams = list(union)
    freq1 = np.array([counts1.get(ng, 0) / total1 for ng in all_ngrams])
    freq2 = np.array([counts2.get(ng, 0) / total2 for ng in all_ngrams])
    
    # Jensen-Shannon divergence
    js_div = float(jensenshannon(freq1, freq2, base=2) ** 2)
    
    # Cosine similarity
    norm1 = np.linalg.norm(freq1)
    norm2 = np.linalg.norm(freq2)
    cosine_sim = float(np.dot(freq1, freq2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
    
    # Find most differentiating n-grams
    # Score = (freq_in_text - freq_in_other) / max(freq_in_other, 0.0001)
    diff_scores1: list[tuple[tuple, float, int]] = []
    diff_scores2: list[tuple[tuple, float, int]] = []
    
    for ng in all_ngrams:
        f1 = counts1.get(ng, 0) / total1
        f2 = counts2.get(ng, 0) / total2
        
        # Prefer n-grams that are frequent in one text and rare/absent in other
        if f1 > f2 and counts1.get(ng, 0) >= 3:
            score = (f1 - f2) / max(f2, 0.0001)
            diff_scores1.append((ng, score, counts1.get(ng, 0)))
        elif f2 > f1 and counts2.get(ng, 0) >= 3:
            score = (f2 - f1) / max(f1, 0.0001)
            diff_scores2.append((ng, score, counts2.get(ng, 0)))
    
    # Sort by score and take top-K
    diff_scores1.sort(key=lambda x: x[1], reverse=True)
    diff_scores2.sort(key=lambda x: x[1], reverse=True)
    
    def make_ngram_str(ng_tuple):
        if all(isinstance(t, str) and len(t) == 1 for t in ng_tuple):
            return "".join(ng_tuple)
        return " ".join(ng_tuple)
    
    top_diff1 = [
        NgramStats(
            ngram=make_ngram_str(ng),
            count=count,
            frequency=count / total1,
            rank=i + 1,
        )
        for i, (ng, score, count) in enumerate(diff_scores1[:top_k_diff])
    ]
    
    top_diff2 = [
        NgramStats(
            ngram=make_ngram_str(ng),
            count=count,
            frequency=count / total2,
            rank=i + 1,
        )
        for i, (ng, score, count) in enumerate(diff_scores2[:top_k_diff])
    ]
    
    return NgramComparisonResult(
        n=n,
        shared_ngrams=len(shared),
        unique_to_text1=len(unique1),
        unique_to_text2=len(unique2),
        jaccard_similarity=float(jaccard),
        js_divergence=js_div,
        cosine_similarity=cosine_sim,
        top_diff_text1=top_diff1,
        top_diff_text2=top_diff2,
    )


# ============================================================================
# UNIFIED METRICS PIPELINE
# ============================================================================


@dataclass
class ComprehensiveMetrics:
    """
    Complete set of all available text metrics in one structure.
    
    This is the result of running the full metrics pipeline on a text.
    Organized by category for easy access.
    """
    
    # === BASIC ENTROPY ===
    shannon_entropy: float
    normalized_entropy: float | None
    perplexity: float
    
    # === LEXICAL DIVERSITY ===
    mtld: float | None
    mattr: float | None
    hdd: float | None
    yules_k: float | None
    ttr: float  # Type-Token Ratio (basic)
    
    # === N-GRAM ENTROPY (letters) ===
    letter_bigram_entropy: float | None
    letter_trigram_entropy: float | None
    letter_bigram_conditional: float | None  # H(X2|X1)
    letter_trigram_conditional: float | None  # H(X3|X1X2)
    
    # === N-GRAM COVERAGE ===
    bigram_coverage_top10: float | None
    trigram_coverage_top10: float | None
    bigram_hapax_ratio: float | None
    trigram_hapax_ratio: float | None
    
    # === SYLLABLE ENTROPY (Russian-specific) ===
    syllable_count_entropy: float | None
    syllable_conditional_entropy: float | None
    vowel_pattern_entropy: float | None
    avg_syllables_per_word: float | None
    
    # === BURSTINESS & MEMORY ===
    burstiness_b: float | None  # Basic B = (σ-μ)/(σ+μ)
    burstiness_parameter: float | None  # BP from intervals
    memory_coefficient: float | None  # MC correlation
    kleinberg_bursts: int | None
    kleinberg_burst_ratio: float | None
    
    # === FRACTAL & COMPLEXITY ===
    hurst_exponent: float | None
    zipf_alpha: float | None
    zipf_r_squared: float | None
    zipf_mandelbrot_alpha: float | None
    zipf_mandelbrot_beta: float | None
    compression_ratio: float | None
    
    # === DISTRIBUTION SHAPE ===
    simpson_index: float | None
    gini_simpson: float | None
    gries_dp_norm: float | None
    pierrehumbert_beta: float | None
    
    # === READABILITY ===
    flesch_reading_ease: float | None
    gunning_fog_index: float | None
    avg_sentence_length: float | None
    
    # === METADATA ===
    n_words: int
    n_unique_words: int
    n_letters: int
    n_unique_letters: int


@dataclass 
class AttributionFeatures:
    """
    Optimized feature set for authorship attribution.
    
    These are the metrics that best distinguish between authors
    based on empirical testing. Normalized for ML compatibility.
    """
    
    # Primary features (highest discriminative power)
    mtld_norm: float  # Normalized MTLD
    yules_k_norm: float  # Normalized Yule's K
    burstiness_b: float  # Already in [-1, 1]
    memory_coefficient: float  # Already in [-1, 1]
    
    # Secondary features (moderate discriminative power)
    mattr: float  # Already in [0, 1]
    hdd: float  # Already in [0, 1]
    letter_bigram_conditional: float  # Normalized
    vowel_pattern_entropy_norm: float  # Normalized
    
    # Tertiary features (style indicators)
    avg_sentence_length_norm: float
    avg_syllables_per_word: float
    compression_ratio: float  # Already in [0, 1]
    
    # Raw values for reference
    raw_mtld: float | None
    raw_yules_k: float | None
    
    def to_vector(self) -> list[float]:
        """Convert to feature vector for ML models."""
        return [
            self.mtld_norm,
            self.yules_k_norm,
            self.burstiness_b,
            self.memory_coefficient,
            self.mattr,
            self.hdd,
            self.letter_bigram_conditional,
            self.vowel_pattern_entropy_norm,
            self.avg_sentence_length_norm,
            self.avg_syllables_per_word,
            self.compression_ratio,
        ]
    
    @staticmethod
    def feature_names() -> list[str]:
        """Return feature names for the vector."""
        return [
            "mtld_norm",
            "yules_k_norm", 
            "burstiness_b",
            "memory_coefficient",
            "mattr",
            "hdd",
            "letter_bigram_conditional",
            "vowel_pattern_entropy_norm",
            "avg_sentence_length_norm",
            "avg_syllables_per_word",
            "compression_ratio",
        ]


def calculate_all_metrics(
    tokens: Sequence[str],
    letters: Sequence[str],
    text: str,
    letter_counts: dict[str, int] | None = None,
    alphabet_size: int = 29,
) -> ComprehensiveMetrics:
    """
    Calculate ALL available metrics in a single pipeline.
    
    This is the master function that computes every metric available
    in the library. Use this when you need comprehensive analysis
    and don't want to call individual functions.
    
    Args:
        tokens: List of words/tokens (for lexical metrics)
        letters: List of individual letters (for n-gram entropy)
        text: Original text (for readability, compression)
        letter_counts: Pre-computed letter frequency dict (optional)
        alphabet_size: Size of alphabet for normalization
        
    Returns:
        ComprehensiveMetrics with all computed values
    """
    n_words = len(tokens)
    n_unique_words = len(set(tokens))
    n_letters = len(letters)
    n_unique_letters = len(set(letters)) if letters else 0
    
    # === BASIC ENTROPY ===
    if letter_counts:
        counts_array = np.array(list(letter_counts.values()), dtype=np.int64)
    else:
        from collections import Counter
        letter_freq = Counter(letters)
        counts_array = np.array(list(letter_freq.values()), dtype=np.int64)
    
    shannon_h = _entropy_from_counts(counts_array) if len(counts_array) > 0 else 0.0
    h_max = np.log2(alphabet_size) if alphabet_size > 0 else 1.0
    normalized_h = shannon_h / h_max if h_max > 0 else None
    perplexity = 2 ** shannon_h if shannon_h > 0 else 1.0
    
    # === LEXICAL DIVERSITY ===
    lex_metrics = calculate_lexical_richness_metrics(tokens) if tokens else {}
    mtld = lex_metrics.get("mtld")
    mattr = lex_metrics.get("mattr")
    hdd = lex_metrics.get("hdd")
    yules_k = lex_metrics.get("yules_k")
    ttr = n_unique_words / n_words if n_words > 0 else 0.0
    
    # === N-GRAM ENTROPY (letters) ===
    letter_bigram_h = None
    letter_trigram_h = None
    letter_bigram_cond = None
    letter_trigram_cond = None
    bigram_cov10 = None
    trigram_cov10 = None
    bigram_hapax = None
    trigram_hapax = None
    
    if len(letters) >= 2:
        bigram_dist = calculate_ngram_distribution(list(letters), n=2, top_k=20)
        if bigram_dist:
            letter_bigram_h = bigram_dist.entropy
            letter_bigram_cond = bigram_dist.conditional_entropy
            bigram_cov10 = bigram_dist.coverage_top_10
            bigram_hapax = bigram_dist.hapax_ratio
    
    if len(letters) >= 3:
        trigram_dist = calculate_ngram_distribution(list(letters), n=3, top_k=20)
        if trigram_dist:
            letter_trigram_h = trigram_dist.entropy
            letter_trigram_cond = trigram_dist.conditional_entropy
            trigram_cov10 = trigram_dist.coverage_top_10
            trigram_hapax = trigram_dist.hapax_ratio
    
    # === SYLLABLE ENTROPY ===
    syllable_result = calculate_syllable_entropy(tokens) if tokens else None
    syllable_count_h = syllable_result.syllable_count_entropy if syllable_result else None
    syllable_cond_h = syllable_result.syllable_count_conditional if syllable_result else None
    vowel_pattern_h = syllable_result.vowel_pattern_entropy if syllable_result else None
    avg_syl = syllable_result.avg_syllables_per_word if syllable_result else None
    
    # === BURSTINESS & MEMORY ===
    burst_metrics = calculate_advanced_burstiness_metrics(tokens) if tokens else None
    burstiness_b = burst_metrics.burstiness_b if burst_metrics else None
    bp = burst_metrics.burstiness_parameter if burst_metrics else None
    mc = burst_metrics.memory_coefficient if burst_metrics else None
    kb = burst_metrics.kleinberg_bursts if burst_metrics else None
    kb_ratio = burst_metrics.kleinberg_burst_ratio if burst_metrics else None
    
    # === FRACTAL & COMPLEXITY ===
    hurst = None
    if tokens and len(tokens) >= 50:
        hurst_result = calculate_hurst_from_word_lengths(tokens)
        hurst = hurst_result.hurst_exponent if hurst_result else None
    
    zipf = calculate_zipf_coefficient(counts_array) if len(counts_array) > 0 else None
    zipf_alpha = zipf.alpha if zipf else None
    zipf_r2 = zipf.r_squared if zipf else None
    
    # Zipf-Mandelbrot on WORD frequencies
    zm_alpha = None
    zm_beta = None
    if tokens:
        from collections import Counter
        word_freq = Counter(tokens)
        word_counts = np.array(sorted(word_freq.values(), reverse=True), dtype=np.int64)
        if len(word_counts) >= 10:
            zm = calculate_zipf_mandelbrot(word_counts)
            zm_alpha = zm.alpha
            zm_beta = zm.beta
    
    compression = calculate_compression_complexity(text) if text else None
    compression_ratio = compression.compression_ratio if compression else None
    
    # === DISTRIBUTION SHAPE ===
    simpson = calculate_simpson_index(counts_array) if len(counts_array) > 0 else None
    gini_simp = calculate_gini_simpson_index(counts_array) if len(counts_array) > 0 else None
    gries = calculate_average_gries_dp(tokens) if tokens else None
    
    pierrehumbert = None
    if tokens and len(tokens) >= 100:
        pierrehumbert = calculate_average_pierrehumbert_beta(tokens, min_word_freq=5, max_words=30)
    
    # === READABILITY ===
    readability = calculate_readability(tokens, text) if tokens and text else None
    flesch = readability.flesch_reading_ease if readability else None
    fog = readability.gunning_fog_index if readability else None
    avg_sent_len = readability.avg_sentence_length if readability else None
    
    return ComprehensiveMetrics(
        shannon_entropy=shannon_h,
        normalized_entropy=normalized_h,
        perplexity=perplexity,
        mtld=mtld,
        mattr=mattr,
        hdd=hdd,
        yules_k=yules_k,
        ttr=ttr,
        letter_bigram_entropy=letter_bigram_h,
        letter_trigram_entropy=letter_trigram_h,
        letter_bigram_conditional=letter_bigram_cond,
        letter_trigram_conditional=letter_trigram_cond,
        bigram_coverage_top10=bigram_cov10,
        trigram_coverage_top10=trigram_cov10,
        bigram_hapax_ratio=bigram_hapax,
        trigram_hapax_ratio=trigram_hapax,
        syllable_count_entropy=syllable_count_h,
        syllable_conditional_entropy=syllable_cond_h,
        vowel_pattern_entropy=vowel_pattern_h,
        avg_syllables_per_word=avg_syl,
        burstiness_b=burstiness_b,
        burstiness_parameter=bp,
        memory_coefficient=mc,
        kleinberg_bursts=kb,
        kleinberg_burst_ratio=kb_ratio,
        hurst_exponent=hurst,
        zipf_alpha=zipf_alpha,
        zipf_r_squared=zipf_r2,
        zipf_mandelbrot_alpha=zm_alpha,
        zipf_mandelbrot_beta=zm_beta,
        compression_ratio=compression_ratio,
        simpson_index=simpson,
        gini_simpson=gini_simp,
        gries_dp_norm=gries,
        pierrehumbert_beta=pierrehumbert,
        flesch_reading_ease=flesch,
        gunning_fog_index=fog,
        avg_sentence_length=avg_sent_len,
        n_words=n_words,
        n_unique_words=n_unique_words,
        n_letters=n_letters,
        n_unique_letters=n_unique_letters,
    )


def extract_attribution_features(
    metrics: ComprehensiveMetrics,
    mtld_scale: float = 500.0,
    yules_k_scale: float = 200.0,
    vowel_h_scale: float = 10.0,
    sentence_len_scale: float = 30.0,
) -> AttributionFeatures:
    """
    Extract normalized features optimized for authorship attribution.
    
    Selects the most discriminative metrics and normalizes them
    to [0, 1] or [-1, 1] range for ML compatibility.
    
    Args:
        metrics: ComprehensiveMetrics from calculate_all_metrics()
        mtld_scale: Expected max MTLD value for normalization
        yules_k_scale: Expected max Yule's K for normalization
        vowel_h_scale: Expected max vowel pattern entropy
        sentence_len_scale: Expected max sentence length
        
    Returns:
        AttributionFeatures ready for ML models
    """
    # Normalize MTLD (typically 50-500 range)
    mtld_norm = min(1.0, (metrics.mtld or 0.0) / mtld_scale)
    
    # Normalize Yule's K (typically 20-200 range, inverse: lower = more diverse)
    yules_k_norm = min(1.0, (metrics.yules_k or 0.0) / yules_k_scale)
    
    # Burstiness already in [-1, 1]
    burstiness = metrics.burstiness_b if metrics.burstiness_b is not None else 0.0
    
    # Memory coefficient already in [-1, 1]
    memory = metrics.memory_coefficient if metrics.memory_coefficient is not None else 0.0
    
    # MATTR already in [0, 1]
    mattr = metrics.mattr if metrics.mattr is not None else 0.5
    
    # HD-D already in [0, 1]
    hdd = metrics.hdd if metrics.hdd is not None else 0.5
    
    # Letter bigram conditional entropy (normalize by max ~4 bits)
    bigram_cond = min(1.0, (metrics.letter_bigram_conditional or 0.0) / 4.0)
    
    # Vowel pattern entropy (normalize)
    vowel_h_norm = min(1.0, (metrics.vowel_pattern_entropy or 0.0) / vowel_h_scale)
    
    # Average sentence length (normalize)
    sent_len_norm = min(1.0, (metrics.avg_sentence_length or 0.0) / sentence_len_scale)
    
    # Average syllables per word (typically 1.5-3.0, normalize to [0, 1])
    avg_syl = min(1.0, (metrics.avg_syllables_per_word or 2.0) / 4.0)
    
    # Compression ratio already in [0, 1]
    compression = metrics.compression_ratio if metrics.compression_ratio is not None else 0.5
    
    return AttributionFeatures(
        mtld_norm=mtld_norm,
        yules_k_norm=yules_k_norm,
        burstiness_b=burstiness,
        memory_coefficient=memory,
        mattr=mattr,
        hdd=hdd,
        letter_bigram_conditional=bigram_cond,
        vowel_pattern_entropy_norm=vowel_h_norm,
        avg_sentence_length_norm=sent_len_norm,
        avg_syllables_per_word=avg_syl,
        compression_ratio=compression,
        raw_mtld=metrics.mtld,
        raw_yules_k=metrics.yules_k,
    )


def calculate_attribution_distance(
    features1: AttributionFeatures,
    features2: AttributionFeatures,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Calculate stylometric distance between two texts for attribution.
    
    Uses weighted Euclidean distance on normalized features.
    Lower distance = more similar authorship style.
    
    Args:
        features1: Attribution features from first text
        features2: Attribution features from second text
        weights: Optional custom weights for each feature
        
    Returns:
        Distance value (0 = identical, higher = more different)
    """
    if weights is None:
        # Default weights based on discriminative power
        weights = {
            "mtld_norm": 2.0,
            "yules_k_norm": 1.5,
            "burstiness_b": 2.0,
            "memory_coefficient": 1.5,
            "mattr": 1.0,
            "hdd": 1.0,
            "letter_bigram_conditional": 1.0,
            "vowel_pattern_entropy_norm": 1.2,
            "avg_sentence_length_norm": 0.8,
            "avg_syllables_per_word": 0.8,
            "compression_ratio": 0.5,
        }
    
    v1 = features1.to_vector()
    v2 = features2.to_vector()
    names = AttributionFeatures.feature_names()
    
    squared_diff = 0.0
    total_weight = 0.0
    
    for i, name in enumerate(names):
        w = weights.get(name, 1.0)
        squared_diff += w * (v1[i] - v2[i]) ** 2
        total_weight += w
    
    # Normalize by total weight
    return float(np.sqrt(squared_diff / total_weight)) if total_weight > 0 else 0.0


def quick_attribution_score(
    tokens: Sequence[str],
    letters: Sequence[str],
    text: str,
) -> AttributionFeatures:
    """
    Quick pipeline to get attribution features from raw text data.
    
    Convenience function that combines calculate_all_metrics()
    and extract_attribution_features() in one call.
    
    Args:
        tokens: List of words/tokens
        letters: List of individual letters
        text: Original text
        
    Returns:
        AttributionFeatures ready for comparison
    """
    metrics = calculate_all_metrics(tokens, letters, text)
    return extract_attribution_features(metrics)
