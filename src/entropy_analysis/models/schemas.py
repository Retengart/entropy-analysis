"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel, Field


class LetterDistribution(BaseModel):
    """Distribution statistics for a single letter."""

    rank: int = Field(..., description="Position in alphabet (1-based)")
    letter: str = Field(..., description="The letter")
    count: int = Field(..., description="Number of occurrences")
    probability: float = Field(..., description="Probability p_i = count / N")
    p_log2: float = Field(..., description="p_i * log2(p_i) contribution")


class NormalizedEntropyResponse(BaseModel):
    """Normalized entropy metrics."""

    h_normalized: float = Field(..., description="H / log2(alphabet_size), i.e. H / H_max")
    entropy_per_word: float = Field(..., description="H / N (average entropy contribution per word)")
    efficiency: float = Field(..., description="How close to maximum entropy (0-1)")


class BootstrapResponse(BaseModel):
    """Bootstrap confidence interval."""

    estimate: float = Field(..., description="Point estimate")
    std_error: float = Field(..., description="Standard error")
    ci_lower: float = Field(..., description="Lower bound of CI")
    ci_upper: float = Field(..., description="Upper bound of CI")
    confidence_level: float = Field(..., description="Confidence level (e.g., 0.95)")
    n_bootstrap: int = Field(..., description="Number of bootstrap samples")


class EnhancedMetricsResponse(BaseModel):
    """Enhanced metrics for comprehensive text analysis."""

    perplexity: float = Field(..., description="Perplexity (2^H, effective number of categories in distribution)")
    alphabet_utilization: float = Field(..., description="Fraction of alphabet used (0-1)")
    evenness: float = Field(..., description="Pielou's evenness index (0-1)")
    herfindahl_index: float = Field(..., description="Herfindahl-Hirschman Index (concentration)")
    uniformity_distance: float = Field(..., description="Distance from uniform distribution")
    uniqueness_ratio: float = Field(..., description="Unique letters per word")
    redundancy: float = Field(..., description="Redundancy (1 - H/H_max)")
    renyi_0: float = Field(..., description="Rényi entropy H₀ (Hartley entropy)")
    renyi_2: float = Field(..., description="Rényi entropy H₂ (Collision entropy)")
    renyi_inf: float = Field(..., description="Rényi entropy H∞ (Min-entropy)")


class TextAnalysisResponse(BaseModel):
    """Complete analysis result for a single text."""

    # Basic metrics
    n_words: int = Field(..., description="Total number of words")
    n_unique_letters: int = Field(..., description="Number of unique first letters")
    shannon_entropy: float | None = Field(None, description="Shannon entropy in bits")
    mean_rank: float | None = Field(None, description="Mean rank of first letters")
    std_rank: float | None = Field(None, description="Standard deviation of ranks")

    # Letter distribution
    letter_distribution: list[LetterDistribution] = Field(
        default_factory=list, description="Per-letter statistics"
    )

    # Normalized metrics
    normalized_entropy: NormalizedEntropyResponse | None = Field(
        None, description="Normalized entropy metrics"
    )
    miller_madow_entropy: float | None = Field(
        None, description="Bias-corrected entropy (Miller-Madow)"
    )

    # Bootstrap
    bootstrap: BootstrapResponse | None = Field(None, description="Bootstrap confidence interval")

    # Diversity indices
    simpson_index: float | None = Field(
        None, description="Simpson's Index (probability of same letter)"
    )
    gini_simpson_index: float | None = Field(
        None, description="Gini-Simpson Index (probability of different letter)"
    )

    # Zipf
    zipf_alpha: float | None = Field(None, description="Zipf's law exponent")
    zipf_r_squared: float | None = Field(None, description="Zipf fit R²")

    # Complexity
    compression_ratio: float | None = Field(None, description="Compression ratio")

    # Enhanced metrics
    enhanced: EnhancedMetricsResponse | None = Field(
        None, description="Enhanced metrics (perplexity, evenness, Rényi entropies, etc.)"
    )

    # NEW METRICS
    yules_k: float | None = Field(None, description="Yule's K (Lexical Concentration)")
    mtld: float | None = Field(None, description="Measure of Textual Lexical Diversity")
    mattr: float | None = Field(
        None, description="Moving Average Type-Token Ratio (0-1, more stable than MTLD for short texts)"
    )
    gries_dp: float | None = Field(
        None, description="Gries' DP (average normalized Deviation of Proportions, 0-1, where 1 = perfect dispersion)"
    )
    bigram_entropy: float | None = Field(None, description="Conditional entropy of bigrams (bits)")
    trigram_entropy: float | None = Field(None, description="Conditional entropy of trigrams (bits)")
    burstiness: float | None = Field(None, description="Burstiness index (B)")
    
    # PHASE 2 METRICS
    hurst_exponent: float | None = Field(
        None, description="Hurst exponent H from DFA (0-1, 0.5=random, >0.5=persistent/fractal, <0.5=anti-persistent)"
    )
    zipf_mandelbrot_alpha: float | None = Field(
        None, description="Zipf-Mandelbrot exponent α (slope parameter)"
    )
    zipf_mandelbrot_beta: float | None = Field(
        None, description="Zipf-Mandelbrot parameter β (curvature correction)"
    )
    pierrehumbert_beta: float | None = Field(
        None, description="Pierrehumbert's β (average Weibull shape parameter, <1=bursty, ≈1=Poisson, >1=regular)"
    )

    # READABILITY METRICS
    flesch_reading_ease: float | None = Field(
        None, 
        description="Flesch Reading Ease (0-100+, higher=easier). 90-100: very easy, 60-70: medium, 30-50: difficult, 0-30: very difficult"
    )
    gunning_fog_index: float | None = Field(
        None, 
        description="Gunning Fog Index (grade level). <6: easy, 7-12: readable, 13-17: medium, 18-24: difficult, >24: very difficult"
    )
    avg_sentence_length: float | None = Field(
        None, description="Average number of words per sentence"
    )
    avg_syllables_per_word: float | None = Field(
        None, description="Average number of syllables per word"
    )

    # Metadata
    source_name: str | None = Field(None, description="Name/path of the source")
    alphabet_size: int = Field(28, description="Size of the alphabet used")


class ExtendedStatsResponse(BaseModel):
    """Extended statistical metrics."""

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


class CorrelationResult(BaseModel):
    """Correlation analysis result."""

    n_samples: int = Field(..., description="Number of data points")
    correlation: float = Field(..., description="Pearson correlation coefficient")
    slope: float = Field(..., description="Linear regression slope")
    intercept: float = Field(..., description="Linear regression intercept")
    r_squared: float = Field(..., description="Coefficient of determination")
    p_value: float = Field(..., description="P-value for correlation significance")


class BatchAnalysisResponse(BaseModel):
    """Result of analyzing multiple texts."""

    results: list[TextAnalysisResponse] = Field(
        default_factory=list, description="Individual text results"
    )
    extended_stats: ExtendedStatsResponse | None = Field(None, description="Aggregate statistics")
    correlation: CorrelationResult | None = Field(None, description="N vs H correlation")


class ComparisonResponse(BaseModel):
    """Result of comparing two texts."""

    text1_name: str = Field(..., description="Name of first text")
    text2_name: str = Field(..., description="Name of second text")
    kl_divergence_p_q: float = Field(..., description="KL(P || Q)")
    kl_divergence_q_p: float = Field(..., description="KL(Q || P)")
    js_divergence: float = Field(..., description="Jensen-Shannon divergence (symmetric)")
    cosine_similarity: float | None = Field(None, description="Cosine similarity")


class RollingEntropyResponse(BaseModel):
    """Rolling entropy analysis result."""

    positions: list[int] = Field(..., description="Window start positions")
    entropies: list[float] = Field(..., description="Entropy at each position")
    window_size: int = Field(..., description="Window size in words")
    mean_entropy: float
    std_entropy: float
    min_entropy: float
    max_entropy: float


# Request models


class UploadedTextRequest(BaseModel):
    """Request for analyzing uploaded text."""

    text: str = Field(..., description="Text content to analyze")
    name: str | None = Field(None, description="Optional name for the text")
    include_bootstrap: bool = Field(False, description="Include bootstrap CI")
    bootstrap_iterations: int = Field(1000, description="Number of bootstrap iterations")
    include_complexity: bool = Field(False, description="Include compression complexity")
    include_advanced_metrics: bool = Field(
        False,
        description="Include advanced Phase 2 metrics (Hurst, Zipf-Mandelbrot, Pierrehumbert beta). "
        "These are computationally expensive and disabled by default.",
    )


class CompareTextsRequest(BaseModel):
    """Request for comparing two texts."""

    text1: str = Field(..., description="First text")
    text2: str = Field(..., description="Second text")
    name1: str = Field("Text 1", description="Name for first text")
    name2: str = Field("Text 2", description="Name for second text")


class BatchTextsRequest(BaseModel):
    """Request for batch analysis."""

    texts: list[UploadedTextRequest] = Field(..., description="List of texts to analyze")
    include_bootstrap: bool = Field(False, description="Include bootstrap CI for each")


class RollingEntropyRequest(BaseModel):
    """Request for rolling entropy analysis."""

    text: str = Field(..., description="Text to analyze")
    window_size: int = Field(50, description="Window size in words")
    step_size: int = Field(10, description="Step size between windows")


class AnalyzerConfigRequest(BaseModel):
    """Configuration for the analyzer."""

    alphabet: str = Field("rus29", description="Alphabet type: rus29, rus33, custom (rus28 is supported for backward compatibility)")
    custom_letters: str | None = Field(None, description="Custom alphabet letters")
    keep_yo: bool = Field(False, description="Keep ё instead of replacing with е")
    keep_j: bool = Field(False, description="Keep й instead of replacing with и")
    min_token_len: int = Field(1, description="Minimum token length")


# Recognition algorithm (lab 40)


class NoiseSettings(BaseModel):
    """Noise injection settings."""

    factor: float = Field(0.0, ge=0, description="Uniform noise magnitude [-factor, factor]")
    mode: str = Field("uniform", description="Noise mode (only 'uniform' supported)")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    renormalize: bool = Field(
        True, description="Renormalize conditional probabilities after noise injection"
    )


class FeatureProbabilityInput(BaseModel):
    """Conditional probabilities for a single feature."""

    name: str | None = Field(None, description="Feature name")
    values_count: int = Field(..., ge=2, description="Number of discrete values for the feature")
    conditional: list[list[float]] = Field(
        ...,
        description="Matrix of shape (M, values_count): P(x_l^k | A_i) per class and value",
    )


class RecognitionRequest(BaseModel):
    """Request to run the recognition algorithm."""

    classes: list[str] = Field(..., description="Class labels A_i (only length is used)")
    priors: list[float] = Field(..., description="Prior probabilities P(A_i)")
    features: list[FeatureProbabilityInput] = Field(
        ..., description="Per-feature conditional probabilities"
    )
    error_target: float = Field(0.05, gt=0, lt=1, description="Target error threshold P(e)_zad")
    noise: NoiseSettings | None = Field(None, description="Noise configuration (optional)")
    use_sample: bool = Field(
        False,
        description="Use built-in lab sample (overrides priors/features); noise still applies if set",
    )


class FeatureMetricsResponse(BaseModel):
    """Per-feature metrics."""

    index: int
    name: str
    values_count: int
    informativeness: float
    error: float
    px: list[float]
    passes_threshold: bool


class PairMetricsResponse(BaseModel):
    """Metrics for a feature pair."""

    indices: tuple[int, int]
    names: tuple[str, str]
    error: float
    passes_threshold: bool


class RecognitionRunResponse(BaseModel):
    """Results for one run (clean or noisy)."""

    features: list[FeatureMetricsResponse]
    best_feature: FeatureMetricsResponse | None
    best_pair: PairMetricsResponse | None
    min_error: float


class RecognitionResponse(BaseModel):
    """Full recognition analysis response."""

    clean: RecognitionRunResponse
    noisy: RecognitionRunResponse | None
    delta_abs: float | None
    delta_rel: float | None


# Batch recognition on text segments


class TextSegment(BaseModel):
    """Text segment with optional label."""

    text: str = Field(..., description="Segment text")
    name: str | None = Field(None, description="Optional name/id")
    label: str | None = Field(None, description="Class label (required for training)")


class SegmentPredictionResponse(BaseModel):
    """Prediction for a single segment."""

    name: str | None
    true_label: str | None
    predicted_label: str
    posteriors: dict[str, float]
    feature_values: dict[str, str]


class TrainedTablesResponse(BaseModel):
    """Trained priors and conditional tables."""

    classes: list[str]
    priors: list[float]
    feature_values: dict[str, list[str]]
    conditionals: dict[str, list[list[float]]]


class RecognitionBatchRequest(BaseModel):
    """Batch recognition request for text segments."""

    segments: list[TextSegment] = Field(..., description="Segments with labels for training")
    error_target: float = Field(0.05, gt=0, lt=1, description="Target error threshold")
    smoothing: float = Field(1e-3, gt=0, description="Laplace smoothing for probabilities")
    noise: NoiseSettings | None = Field(None, description="Optional noise injection")


class RecognitionBatchResponse(BaseModel):
    """Batch recognition response."""

    tables: TrainedTablesResponse
    recognition: RecognitionRunResponse
    predictions: list[SegmentPredictionResponse]
