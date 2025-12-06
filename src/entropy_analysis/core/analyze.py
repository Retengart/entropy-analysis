"""
Main text analyzer module.

Provides the TextAnalyzer class that orchestrates all analysis operations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from entropy_analysis.core.attribution import (
    calculate_burrows_delta,
    calculate_cosine_delta,
    calculate_eders_delta,
    calculate_z_scores,
)
from entropy_analysis.core.metrics import (
    calculate_average_gries_dp,
    calculate_average_pierrehumbert_beta,
    calculate_burstiness_index,
    calculate_compression_complexity,
    calculate_gini_simpson_index,
    calculate_hurst_from_word_lengths,
    calculate_js_divergence,
    calculate_kl_divergence,
    calculate_mattr,
    calculate_mtld,
    calculate_ngram_entropy,
    calculate_readability,
    calculate_rolling_entropy,
    calculate_simpson_index,
    calculate_yules_k,
    calculate_zipf_coefficient,
    calculate_zipf_mandelbrot,
)
from entropy_analysis.core.normalize import (
    Alphabet,
    Normalizer,
    create_normalizer,
)
from entropy_analysis.core.stats import (
    BootstrapResult,
    EnhancedMetrics,
    ExtendedStats,
    NormalizedEntropy,
    bootstrap_entropy_confidence,
    calculate_correlation,
    calculate_information_distances,
    calculate_enhanced_metrics,
    calculate_extended_stats,
    calculate_mad,
    calculate_miller_madow_correction,
    calculate_normalized_entropy,
    calculate_shannon_entropy,
    detect_outlier_iqr,
    detect_outlier_modified_zscore,
    detect_outlier_zscore,
)


@dataclass
class LetterStats:
    """Statistics for a single letter."""

    rank: int
    letter: str
    count: int
    probability: float
    p_log2: float  # p * log2(p) contribution to entropy


@dataclass
class TextAnalysisResult:
    """Complete analysis result for a single text."""

    # Basic metrics
    n_words: int
    n_unique_letters: int
    shannon_entropy: float | None
    mean_rank: float | None
    std_rank: float | None

    # Letter distribution
    letter_stats: list[LetterStats]

    # Normalized metrics
    normalized_entropy: NormalizedEntropy | None = None
    miller_madow_entropy: float | None = None

    # Bootstrap confidence interval
    bootstrap: BootstrapResult | None = None

    # Diversity indices
    simpson_index: float | None = None
    gini_simpson_index: float | None = None

    # Zipf analysis
    zipf_alpha: float | None = None
    zipf_r_squared: float | None = None

    # Complexity
    compression_ratio: float | None = None

    # Enhanced metrics
    enhanced: EnhancedMetrics | None = None
    
    # NEW METRICS
    yules_k: float | None = None
    mtld: float | None = None
    mattr: float | None = None
    gries_dp: float | None = None  # Average normalized DP
    bigram_entropy: float | None = None
    trigram_entropy: float | None = None
    burstiness: float | None = None
    
    # PHASE 2 METRICS
    hurst_exponent: float | None = None  # Hurst H from DFA
    zipf_mandelbrot_alpha: float | None = None  # Zipf-Mandelbrot α
    zipf_mandelbrot_beta: float | None = None  # Zipf-Mandelbrot β
    pierrehumbert_beta: float | None = None  # Average Weibull β

    # READABILITY METRICS
    flesch_reading_ease: float | None = None  # Flesch Reading Ease (0-100+)
    gunning_fog_index: float | None = None  # Gunning Fog Index (grade level)
    avg_sentence_length: float | None = None  # Average words per sentence
    avg_syllables_per_word: float | None = None  # Average syllables per word

    # Metadata
    source_name: str | None = None
    alphabet_size: int = 29


@dataclass
class BatchAnalysisResult:
    """Result of analyzing multiple texts."""

    results: list[tuple[str, TextAnalysisResult]]
    extended_stats: ExtendedStats | None = None

    # Correlation between N and H
    correlation: float | None = None
    correlation_slope: float | None = None
    correlation_intercept: float | None = None
    correlation_r_squared: float | None = None
    correlation_p_value: float | None = None


@dataclass
class ComparisonResult:
    """Result of comparing two texts or distributions."""

    text1_name: str
    text2_name: str
    kl_divergence_p_q: float  # KL(P || Q)
    kl_divergence_q_p: float  # KL(Q || P)
    js_divergence: float  # Jensen-Shannon (symmetric)
    cosine_similarity: float | None = None
    
    # Information distances (more robust metrics)
    wasserstein_distance: float | None = None
    hellinger_distance: float | None = None
    total_variation_distance: float | None = None
    
    # Delta metrics (Attribution)
    burrows_delta: float | None = None
    eders_delta: float | None = None
    cosine_delta: float | None = None
    mfw_used: int | None = None


@dataclass
class TextAnalyzer:
    """
    Main analyzer class for entropy analysis of texts.

    Provides methods for:
    - Single text analysis
    - Batch analysis of multiple texts
    - Comparison between texts
    - Rolling entropy analysis
    """

    normalizer: Normalizer = field(default_factory=create_normalizer)

    @classmethod
    def create(
        cls,
        alphabet: Alphabet | str = Alphabet.RUS29,
        custom_letters: str | None = None,
        keep_yo: bool = False,
        keep_j: bool = False,
        min_token_len: int = 1,
    ) -> "TextAnalyzer":
        """Factory method to create analyzer with custom configuration."""
        normalizer = create_normalizer(
            alphabet=alphabet,
            custom_letters=custom_letters,
            keep_yo=keep_yo,
            keep_j=keep_j,
            min_token_len=min_token_len,
        )
        return cls(normalizer=normalizer)

    def analyze(
        self,
        text: str,
        source_name: str | None = None,
        include_bootstrap: bool = False,
        bootstrap_iterations: int = 1000,
        include_complexity: bool = False,
        include_advanced_metrics: bool = False,
        log_callback: Callable[[str], None] | None = None,
    ) -> TextAnalysisResult:
        """
        Analyze a single text and return comprehensive metrics.

        Args:
            text: Input text to analyze
            source_name: Optional name/identifier for the text
            include_bootstrap: Whether to calculate bootstrap confidence intervals
            bootstrap_iterations: Number of bootstrap iterations
            include_complexity: Whether to calculate compression complexity
            include_advanced_metrics: Whether to calculate advanced Phase 2 metrics
            log_callback: Optional function to receive progress log messages

        Returns:
            TextAnalysisResult with all metrics
        """
        def log(msg: str):
            if log_callback:
                log_callback(msg)

        log(f"Начинаем анализ текста: {source_name or 'unnamed'} ({len(text)} символов)...")

        # Count first letters
        letter_counts = self.normalizer.count_first_letters(text)
        n_words = sum(letter_counts.values())

        # Build counts array for all letters in alphabet
        counts_array = np.array(
            [letter_counts.get(letter, 0) for letter in self.normalizer.letters],
            dtype=np.int64,
        )

        # Build letter stats
        letter_stats: list[LetterStats] = []
        for i, letter in enumerate(self.normalizer.letters):
            count = counts_array[i]
            p = count / n_words if n_words > 0 else 0.0
            p_log2 = p * np.log2(p) if p > 0 else 0.0
            letter_stats.append(
                LetterStats(
                    rank=i + 1,
                    letter=letter,
                    count=int(count),
                    probability=p,
                    p_log2=p_log2,
                )
            )

        # Initialize metrics
        shannon_h = None
        n_unique = 0
        mean_rank = None
        std_rank = None
        normalized = None
        mm_entropy = None
        simpson = None
        gini_simpson = None
        zipf = None
        bootstrap_result = None
        compression_ratio = None
        enhanced = None
        
        yules_k = None
        mtld = None
        mattr = None
        gries_dp = None
        bigram_h = None
        trigram_h = None
        burstiness = None
        
        # Phase 2 metrics
        hurst_result = None
        zipf_mandelbrot = None
        pierrehumbert_beta = None
        
        # Readability metrics
        readability = None

        # Calculate metrics if we have words
        if n_words > 0:
            shannon_h = calculate_shannon_entropy(counts_array)
            n_unique = int(np.sum(counts_array > 0))

            # Mean and std of ranks
            probs = counts_array / n_words
            ranks = np.arange(1, len(counts_array) + 1, dtype=np.float64)
            mean_rank = float(np.sum(ranks * probs))
            var_rank = float(np.sum((ranks - mean_rank) ** 2 * probs))
            std_rank = float(np.sqrt(var_rank))

            # Normalized entropy
            normalized = calculate_normalized_entropy(
                shannon_h, self.normalizer.alphabet_size, n_words
            )

            # Miller-Madow correction
            mm_entropy = calculate_miller_madow_correction(shannon_h, n_unique, n_words)

            # Diversity indices
            simpson = calculate_simpson_index(counts_array)
            gini_simpson = calculate_gini_simpson_index(counts_array)

            # Zipf analysis
            zipf = calculate_zipf_coefficient(counts_array)

            # Bootstrap (optional, computationally expensive)
            if include_bootstrap:
                log(f"Запуск Bootstrap анализа ({bootstrap_iterations} итераций)...")
                bootstrap_result = bootstrap_entropy_confidence(
                    counts_array, n_bootstrap=bootstrap_iterations
                )

            # Compression complexity (optional)
            if include_complexity:
                log("Расчет алгоритмической сложности (сжатие)...")
                complexity = calculate_compression_complexity(text)
                compression_ratio = complexity.compression_ratio

            # Enhanced metrics
            enhanced = calculate_enhanced_metrics(
                counts_array,
                n_words,
                self.normalizer.alphabet_size,
                shannon_h,
            )
            
            # New Word-based metrics
            log("Токенизация и расчет лексических метрик...")
            tokens = self.normalizer.tokenize(text)
            if tokens:
                # Readability indices (Flesch & Gunning Fog)
                log("Расчет индексов удобочитаемости...")
                readability = calculate_readability(tokens, text)
                
                yules_k = calculate_yules_k(tokens)
                mtld = calculate_mtld(tokens)
                # MATTR: use smaller window for shorter texts
                mattr_window = min(500, max(50, len(tokens) // 10))
                mattr = calculate_mattr(tokens, window_size=mattr_window)
                # Gries' DP: average normalized DP across all words
                gries_dp = calculate_average_gries_dp(tokens, n_segments=10, min_word_freq=3)
                bigram_h = calculate_ngram_entropy(tokens, n=2)
                trigram_h = calculate_ngram_entropy(tokens, n=3)
                burstiness = calculate_burstiness_index(tokens)
                
                # Phase 2: Advanced metrics (optional, computationally expensive)
                if include_advanced_metrics:
                    log("Расчет продвинутых метрик (Hurst, Zipf-Mandelbrot, Pierrehumbert)...")
                    
                    # Hurst exponent from word lengths
                    if len(tokens) >= 50:  # Need sufficient data for DFA
                        log("...Hurst Exponent (DFA)")
                        hurst_result = calculate_hurst_from_word_lengths(tokens)
                    
                    # Zipf-Mandelbrot (improved Zipf)
                    log("...Zipf-Mandelbrot fitting")
                    zipf_mandelbrot = calculate_zipf_mandelbrot(counts_array)
                    
                    # Pierrehumbert's beta (Weibull fitting) - ограничиваем количество слов
                    if len(tokens) >= 100:  # Need sufficient data
                        log("...Pierrehumbert's Beta (Weibull fitting)")
                        pierrehumbert_beta = calculate_average_pierrehumbert_beta(
                            tokens,
                            min_word_freq=max(5, len(tokens) // 100),
                            max_words=30,  # Ограничиваем топ-30 словами для ускорения
                        )
        
        log("Анализ завершен.")
        return TextAnalysisResult(
            n_words=n_words,
            n_unique_letters=n_unique,
            shannon_entropy=shannon_h,
            mean_rank=mean_rank,
            std_rank=std_rank,
            letter_stats=letter_stats,
            normalized_entropy=normalized,
            miller_madow_entropy=mm_entropy,
            bootstrap=bootstrap_result,
            simpson_index=simpson,
            gini_simpson_index=gini_simpson,
            zipf_alpha=zipf.alpha if zipf else None,
            zipf_r_squared=zipf.r_squared if zipf else None,
            compression_ratio=compression_ratio,
            enhanced=enhanced,
            yules_k=yules_k,
            mtld=mtld,
            mattr=mattr,
            gries_dp=gries_dp,
            bigram_entropy=bigram_h,
            trigram_entropy=trigram_h,
            burstiness=burstiness,
            hurst_exponent=hurst_result.hurst_exponent if hurst_result else None,
            zipf_mandelbrot_alpha=zipf_mandelbrot.alpha if zipf_mandelbrot else None,
            zipf_mandelbrot_beta=zipf_mandelbrot.beta if zipf_mandelbrot else None,
            pierrehumbert_beta=pierrehumbert_beta,
            flesch_reading_ease=readability.flesch_reading_ease if readability else None,
            gunning_fog_index=readability.gunning_fog_index if readability else None,
            avg_sentence_length=readability.avg_sentence_length if readability else None,
            avg_syllables_per_word=readability.avg_syllables_per_word if readability else None,
            source_name=source_name,
            alphabet_size=self.normalizer.alphabet_size,
        )

    def analyze_batch(
        self,
        texts: list[tuple[str, str]],
        include_bootstrap: bool = False,
        include_advanced_metrics: bool = False,
        log_callback: Callable[[str], None] | None = None,
    ) -> BatchAnalysisResult:
        """
        Analyze multiple texts and calculate aggregate statistics.

        Args:
            texts: List of (name, text) tuples
            include_bootstrap: Whether to calculate bootstrap CI for each text
            include_advanced_metrics: Whether to calculate advanced Phase 2 metrics
            log_callback: Optional logger

        Returns:
            BatchAnalysisResult with all results and aggregate stats
        """
        results: list[tuple[str, TextAnalysisResult]] = []

        for i, (name, text) in enumerate(texts):
            if log_callback:
                log_callback(f"[{i+1}/{len(texts)}] Обработка: {name}")
            
            result = self.analyze(
                text,
                source_name=name,
                include_bootstrap=include_bootstrap,
                include_advanced_metrics=include_advanced_metrics,
                log_callback=log_callback if len(texts) == 1 else None, # Pass down only for single items or handle differently
            )
            results.append((name, result))

        # Calculate extended statistics on entropy values
        entropies = [r.shannon_entropy for _, r in results if r.shannon_entropy is not None]
        extended = calculate_extended_stats(entropies) if len(entropies) >= 2 else None

        # Calculate correlation between N and H
        n_values = [r.n_words for _, r in results if r.shannon_entropy is not None]
        h_values = [r.shannon_entropy for _, r in results if r.shannon_entropy is not None]

        corr_result = None
        if len(n_values) >= 2:
            corr_result = calculate_correlation(n_values, h_values)

        return BatchAnalysisResult(
            results=results,
            extended_stats=extended,
            correlation=corr_result[0] if corr_result else None,
            correlation_slope=corr_result[1] if corr_result else None,
            correlation_intercept=corr_result[2] if corr_result else None,
            correlation_r_squared=corr_result[3] if corr_result else None,
            correlation_p_value=corr_result[4] if corr_result else None,
        )

    def analyze_directory(
        self,
        directory: str | Path,
        pattern: str = "*.txt",
        recursive: bool = True,
        include_bootstrap: bool = False,
    ) -> BatchAnalysisResult:
        """
        Analyze all text files in a directory.

        Args:
            directory: Path to directory
            pattern: Glob pattern for files
            recursive: Whether to search subdirectories
            include_bootstrap: Whether to calculate bootstrap CI

        Returns:
            BatchAnalysisResult with all results
        """
        dir_path = Path(directory)
        glob_method = dir_path.rglob if recursive else dir_path.glob

        texts: list[tuple[str, str]] = []
        for file_path in glob_method(pattern):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    texts.append((str(file_path), content))
                except Exception:
                    continue

        return self.analyze_batch(texts, include_bootstrap=include_bootstrap)

    def compare(
        self,
        text1: str,
        text2: str,
        name1: str = "Text 1",
        name2: str = "Text 2",
        include_delta: bool = False,
        mfw_limit: int = 100,
    ) -> ComparisonResult:
        """
        Compare two texts using divergence metrics and optionally Delta.

        Args:
            text1: First text
            text2: Second text
            name1: Name for first text
            name2: Name for second text
            include_delta: Whether to calculate Delta metrics (Burrows, Eder, Cosine)
            mfw_limit: Number of Most Frequent Words to use for Delta

        Returns:
            ComparisonResult with divergence metrics
        """
        # Get letter distributions
        counts1 = self.normalizer.count_first_letters(text1)
        counts2 = self.normalizer.count_first_letters(text2)

        # Build probability arrays
        n1 = sum(counts1.values())
        n2 = sum(counts2.values())

        if n1 == 0 or n2 == 0:
            return ComparisonResult(
                text1_name=name1,
                text2_name=name2,
                kl_divergence_p_q=float("inf"),
                kl_divergence_q_p=float("inf"),
                js_divergence=1.0,
                cosine_similarity=0.0,
            )

        probs1 = np.array(
            [counts1.get(letter, 0) / n1 for letter in self.normalizer.letters],
            dtype=np.float64,
        )
        probs2 = np.array(
            [counts2.get(letter, 0) / n2 for letter in self.normalizer.letters],
            dtype=np.float64,
        )

        # Calculate divergences
        kl_p_q = calculate_kl_divergence(probs1, probs2)
        kl_q_p = calculate_kl_divergence(probs2, probs1)
        js = calculate_js_divergence(probs1, probs2)

        # Cosine similarity (letters)
        norm1 = np.linalg.norm(probs1)
        norm2 = np.linalg.norm(probs2)
        cosine_sim = (
            float(np.dot(probs1, probs2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
        )
        
        # Information distances (more robust for comparing distributions)
        info_dist = calculate_information_distances(probs1, probs2)
        
        # Delta Metrics (Attribution)
        burrows = None
        eders = None
        cosine_d = None
        
        if include_delta:
            # 1. Tokenize
            tokens1 = self.normalizer.tokenize(text1)
            tokens2 = self.normalizer.tokenize(text2)
            
            if tokens1 and tokens2:
                # 2. Build vocabulary (MFW from combined text)
                from collections import Counter
                
                combined_counts = Counter(tokens1) + Counter(tokens2)
                mfw = [word for word, _ in combined_counts.most_common(mfw_limit)]
                
                if mfw:
                    # 3. Create frequency vectors
                    # Relative frequencies (counts / total)
                    # Note: Z-score calculation usually requires a reference CORPUS.
                    # Here we treat the two texts as the "corpus" for standardization.
                    # This is a simplification for pair-wise comparison.
                    
                    freqs1 = np.array([tokens1.count(w) / len(tokens1) for w in mfw])
                    freqs2 = np.array([tokens2.count(w) / len(tokens2) for w in mfw])
                    
                    # Stack for Z-score calculation
                    all_freqs = np.vstack([freqs1, freqs2])
                    
                    # Calculate means and stds across the two texts
                    means = np.mean(all_freqs, axis=0)
                    stds = np.std(all_freqs, axis=0, ddof=1) # Sample std
                    
                    # Calculate Z-scores
                    z1 = calculate_z_scores(freqs1, means, stds)
                    z2 = calculate_z_scores(freqs2, means, stds)
                    
                    # Calculate Deltas
                    burrows = calculate_burrows_delta(z1, z2)
                    eders = calculate_eders_delta(z1, z2)
                    cosine_d = calculate_cosine_delta(z1, z2)

        return ComparisonResult(
            text1_name=name1,
            text2_name=name2,
            kl_divergence_p_q=kl_p_q.divergence,
            kl_divergence_q_p=kl_q_p.divergence,
            js_divergence=js.divergence,
            cosine_similarity=cosine_sim,
            wasserstein_distance=info_dist.wasserstein,
            hellinger_distance=info_dist.hellinger,
            total_variation_distance=info_dist.total_variation,
            burrows_delta=burrows,
            eders_delta=eders,
            cosine_delta=cosine_d,
            mfw_used=mfw_limit if include_delta else None,
        )

    def rolling_entropy(
        self,
        text: str,
        window_size: int = 50,
        step_size: int = 10,
    ):
        """
        Calculate rolling entropy through the text.

        Args:
            text: Input text
            window_size: Number of words per window
            step_size: Step between windows

        Returns:
            RollingEntropyResult
        """
        # Extract first letters as sequence
        first_letters = list(self.normalizer.extract_first_letters(text))

        return calculate_rolling_entropy(
            first_letters,
            window_size=window_size,
            step_size=step_size,
            alphabet=self.normalizer.letters,
        )

    def to_dataframe(self, result: TextAnalysisResult) -> pl.DataFrame:
        """
        Convert letter stats to Polars DataFrame.

        Args:
            result: Analysis result

        Returns:
            Polars DataFrame with letter statistics
        """
        return pl.DataFrame(
            {
                "rank": [s.rank for s in result.letter_stats],
                "letter": [s.letter for s in result.letter_stats],
                "count": [s.count for s in result.letter_stats],
                "probability": [s.probability for s in result.letter_stats],
                "p_log2": [s.p_log2 for s in result.letter_stats],
            }
        )

    def batch_to_dataframe(self, batch: BatchAnalysisResult) -> pl.DataFrame:
        """
        Convert batch results to Polars DataFrame.

        Args:
            batch: Batch analysis result

        Returns:
            Polars DataFrame with summary for each text
        """
        data: dict[str, list[Any]] = {
            "name": [],
            "n_words": [],
            "n_unique_letters": [],
            "shannon_entropy": [],
            "mean_rank": [],
            "std_rank": [],
            "simpson_index": [],
            "zipf_alpha": [],
            "yules_k": [],
            "mtld": [],
            "mattr": [],
            "gries_dp": [],
            "bigram_entropy": [],
            "trigram_entropy": [],
            "burstiness": [],
            "hurst_exponent": [],
            "zipf_mandelbrot_alpha": [],
            "zipf_mandelbrot_beta": [],
            "pierrehumbert_beta": [],
            "flesch_reading_ease": [],
            "gunning_fog_index": [],
        }

        for name, result in batch.results:
            data["name"].append(name)
            data["n_words"].append(result.n_words)
            data["n_unique_letters"].append(result.n_unique_letters)
            data["shannon_entropy"].append(result.shannon_entropy)
            data["mean_rank"].append(result.mean_rank)
            data["std_rank"].append(result.std_rank)
            data["simpson_index"].append(result.simpson_index)
            data["zipf_alpha"].append(result.zipf_alpha)
            data["yules_k"].append(result.yules_k)
            data["mtld"].append(result.mtld)
            data["mattr"].append(result.mattr)
            data["gries_dp"].append(result.gries_dp)
            data["bigram_entropy"].append(result.bigram_entropy)
            data["trigram_entropy"].append(result.trigram_entropy)
            data["burstiness"].append(result.burstiness)
            data["hurst_exponent"].append(result.hurst_exponent)
            data["zipf_mandelbrot_alpha"].append(result.zipf_mandelbrot_alpha)
            data["zipf_mandelbrot_beta"].append(result.zipf_mandelbrot_beta)
            data["pierrehumbert_beta"].append(result.pierrehumbert_beta)
            data["flesch_reading_ease"].append(result.flesch_reading_ease)
            data["gunning_fog_index"].append(result.gunning_fog_index)

        return pl.DataFrame(data)

    def detect_outliers(
        self,
        batch: BatchAnalysisResult,
        method: str = "iqr",
        threshold: float | None = None,
    ) -> list[tuple[str, TextAnalysisResult]]:
        """
        Detect outliers in batch results.

        Args:
            batch: Batch analysis result
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Custom threshold (uses defaults if None)

        Returns:
            List of (name, result) tuples that are outliers
        """
        entropies = [r.shannon_entropy for _, r in batch.results if r.shannon_entropy is not None]

        if len(entropies) < 3:
            return []

        extended = calculate_extended_stats(entropies)
        if extended is None:
            return []

        outliers: list[tuple[str, TextAnalysisResult]] = []

        for name, result in batch.results:
            if result.shannon_entropy is None:
                continue

            h = result.shannon_entropy

            if method == "iqr":
                detection = detect_outlier_iqr(
                    h, extended.q1, extended.q3, extended.iqr, threshold or 1.5
                )
            elif method == "zscore":
                detection = detect_outlier_zscore(
                    h, extended.mean, extended.std_dev, threshold or 3.0
                )
            elif method == "modified_zscore":
                mad = calculate_mad(entropies)
                detection = detect_outlier_modified_zscore(
                    h, extended.median, mad, threshold or 3.5
                )
            else:
                raise ValueError(f"Unknown outlier method: {method}")

            if detection.is_outlier:
                outliers.append((name, result))

        return outliers

    def split_and_analyze(
        self,
        text: str,
        delimiter: str = "***",
        auto_name: bool = True,
        include_bootstrap: bool = False,
        include_advanced_metrics: bool = False,
        log_callback: Callable[[str], None] | None = None,
    ) -> BatchAnalysisResult:
        """
        Split text by delimiter and analyze each segment.

        This is useful for analyzing multiple texts in a single file,
        separated by a delimiter (e.g., "***").

        Args:
            text: Input text with segments separated by delimiter
            delimiter: String delimiter to split on
            auto_name: Whether to auto-generate names (Segment 1, Segment 2, etc.)
            include_bootstrap: Whether to calculate bootstrap CI for each segment
            include_advanced_metrics: Whether to calculate advanced Phase 2 metrics
            log_callback: Optional logger

        Returns:
            BatchAnalysisResult with analysis for each segment
        """
        if log_callback:
            log_callback("Разделение текста на сегменты...")
            
        # Split text by delimiter
        segments = text.split(delimiter)

        # Filter out empty segments
        segments = [seg.strip() for seg in segments if seg.strip()]

        if not segments:
            if log_callback:
                log_callback("Нет валидных сегментов для анализа.")
            # Return empty batch result
            return BatchAnalysisResult(
                results=[],
                extended_stats=None,
                correlation=None,
                correlation_slope=None,
                correlation_intercept=None,
                correlation_r_squared=None,
                correlation_p_value=None,
            )

        # Create list of (name, text) tuples
        texts: list[tuple[str, str]] = []
        for i, segment in enumerate(segments, start=1):
            name = f"Segment {i}" if auto_name else f"Text {i}"
            texts.append((name, segment))

        if log_callback:
            log_callback(f"Найдено {len(texts)} сегментов. Начинаем анализ...")

        # Use existing batch analysis
        return self.analyze_batch(
            texts,
            include_bootstrap=include_bootstrap,
            include_advanced_metrics=include_advanced_metrics,
            log_callback=log_callback,
        )
