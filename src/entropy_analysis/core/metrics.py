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
from scipy import stats
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
    # By convention, 0 * log(0/q) = 0
    mask = p_arr > 0
    kl_div = np.sum(p_arr[mask] * np.log(p_arr[mask] / q_smoothed[mask]) / np.log(base))

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

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    context = [tuple(tokens[i : i + n - 1]) for i in range(len(tokens) - n + 1)]

    # H(N-gram)
    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())
    ngram_probs = np.array(list(ngram_counts.values())) / total_ngrams
    h_ngram = -np.sum(ngram_probs * np.log2(ngram_probs))

    # H(Context)
    context_counts = Counter(context)
    total_context = sum(context_counts.values())
    context_probs = np.array(list(context_counts.values())) / total_context
    h_context = -np.sum(context_probs * np.log2(context_probs))

    return float(h_ngram - h_context)


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
) -> ZipfMandelbrotAnalysis:
    """
    Analyze distribution using Zipf-Mandelbrot law.

    Zipf-Mandelbrot: f(r) = C / (r + beta)^alpha

    This is more accurate than simple Zipf's law, especially for the
    most frequent words. The beta parameter corrects curvature at the
    beginning of the distribution.

    Args:
        counts: Array of counts (will be sorted by frequency)

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

    # Ограничиваем количество точек для регрессии (макс 200) для ускорения
    max_points = 200
    if len(sorted_counts) > max_points:
        # Берем равномерно распределенные точки
        indices = np.linspace(0, len(sorted_counts) - 1, max_points, dtype=int)
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
        # Fit using curve_fit (ограничиваем итерации для ускорения)
        popt, _ = curve_fit(
            zipf_mandelbrot_func,
            ranks,
            log_counts,
            p0=[initial_alpha, initial_beta, initial_log_c],
            bounds=([0.1, 0.0, -np.inf], [5.0, 100.0, np.inf]),
            maxfev=1000,  # Уменьшено с 5000 для ускорения
        )

        alpha_fit, beta_fit, log_c_fit = popt
        c_fit = np.exp(log_c_fit)

        # Calculate R²
        predicted_log = zipf_mandelbrot_func(ranks, alpha_fit, beta_fit, log_c_fit)
        ss_res = np.sum((log_counts - predicted_log) ** 2)
        ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
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
