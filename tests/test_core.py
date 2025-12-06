"""Tests for core analysis functionality."""

import pytest

from entropy_analysis.core.analyze import TextAnalyzer
from entropy_analysis.core.metrics import (
    calculate_js_divergence,
    calculate_kl_divergence,
    calculate_simpson_index,
    calculate_zipf_coefficient,
)
from entropy_analysis.core.normalize import (
    create_normalizer,
)
from entropy_analysis.core.stats import (
    bootstrap_entropy_confidence,
    calculate_extended_stats,
    calculate_miller_madow_correction,
    calculate_shannon_entropy,
)


class TestNormalizer:
    """Tests for the Normalizer class."""

    def test_rus29_alphabet(self):
        """Test that rus29 alphabet has 29 letters (33 - ё - й - ъ - ь)."""
        normalizer = create_normalizer(alphabet="rus29")
        assert normalizer.alphabet_size == 29  # RUS29_LETTERS содержит 29 букв
        assert "ё" not in normalizer.letters
        assert "й" not in normalizer.letters
        assert "ъ" not in normalizer.letters
        assert "ь" not in normalizer.letters

    def test_rus33_alphabet(self):
        """Test that rus33 alphabet has 33 letters."""
        normalizer = create_normalizer(alphabet="rus33")
        assert normalizer.alphabet_size == 33
        assert "ё" in normalizer.letters
        assert "й" in normalizer.letters

    def test_normalization_yo_replacement(self):
        """Test that ё is replaced with е by default."""
        normalizer = create_normalizer(alphabet="rus29", keep_yo=False)
        counts = normalizer.count_first_letters("Ёлка ежик")
        assert counts.get("е", 0) == 2
        assert counts.get("ё", 0) == 0

    def test_normalization_keep_yo(self):
        """Test that ё is kept when keep_yo=True."""
        normalizer = create_normalizer(alphabet="rus33", keep_yo=True)
        counts = normalizer.count_first_letters("Ёлка ежик")
        assert counts.get("ё", 0) == 1
        assert counts.get("е", 0) == 1

    def test_tokenization(self):
        """Test basic tokenization."""
        normalizer = create_normalizer()
        tokens = normalizer.tokenize("Мой дядя самых честных правил")
        assert len(tokens) == 5
        assert tokens[0] == "мой"

    def test_first_letters_extraction(self):
        """Test extraction of first letters."""
        normalizer = create_normalizer()
        letters = list(normalizer.extract_first_letters("Мой дядя самых честных правил"))
        assert letters == ["м", "д", "с", "ч", "п"]

    def test_count_first_letters(self):
        """Test counting first letters."""
        normalizer = create_normalizer()
        text = "Мой мир мечты"
        counts = normalizer.count_first_letters(text)
        assert counts["м"] == 3


class TestStats:
    """Tests for statistical functions."""

    def test_shannon_entropy_uniform(self):
        """Test entropy of uniform distribution."""
        # 4 categories with equal counts = log2(4) = 2 bits
        counts = [10, 10, 10, 10]
        h = calculate_shannon_entropy(counts)
        assert abs(h - 2.0) < 1e-10

    def test_shannon_entropy_single(self):
        """Test entropy when all mass is on one category."""
        counts = [100, 0, 0, 0]
        h = calculate_shannon_entropy(counts)
        assert h == 0.0

    def test_shannon_entropy_empty(self):
        """Test entropy of empty counts."""
        counts = [0, 0, 0]
        h = calculate_shannon_entropy(counts)
        assert h == 0.0

    def test_miller_madow_correction(self):
        """Test Miller-Madow bias correction."""
        entropy = 4.0
        n_categories = 10
        n_samples = 100

        corrected = calculate_miller_madow_correction(entropy, n_categories, n_samples)

        # Correction = (K-1)/(2N) = 9/200 = 0.045
        expected = 4.0 + 0.045
        assert abs(corrected - expected) < 1e-10

    def test_extended_stats(self):
        """Test extended statistics calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculate_extended_stats(values)

        assert stats is not None
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min_value == 1.0
        assert stats.max_value == 5.0

    def test_bootstrap_confidence(self):
        """Test bootstrap confidence interval."""
        counts = [100, 80, 60, 40, 20]
        result = bootstrap_entropy_confidence(counts, n_bootstrap=100, random_state=42)

        assert result.estimate > 0
        assert result.ci_lower < result.estimate
        assert result.ci_upper > result.estimate
        assert result.confidence_level == 0.95


class TestMetrics:
    """Tests for advanced metrics."""

    def test_kl_divergence_same_distribution(self):
        """Test KL divergence of identical distributions."""
        p = [0.25, 0.25, 0.25, 0.25]
        q = [0.25, 0.25, 0.25, 0.25]

        result = calculate_kl_divergence(p, q)
        assert result.divergence < 1e-6  # Should be ~0

    def test_kl_divergence_different(self):
        """Test KL divergence of different distributions."""
        p = [0.9, 0.1]
        q = [0.5, 0.5]

        result = calculate_kl_divergence(p, q)
        assert result.divergence > 0
        assert not result.symmetric

    def test_js_divergence_symmetric(self):
        """Test that JS divergence is symmetric."""
        p = [0.7, 0.3]
        q = [0.4, 0.6]

        js_pq = calculate_js_divergence(p, q)
        js_qp = calculate_js_divergence(q, p)

        assert abs(js_pq.divergence - js_qp.divergence) < 1e-10
        assert js_pq.symmetric

    def test_simpson_index_uniform(self):
        """Test Simpson index for uniform distribution."""
        # For uniform: D = 1/N
        counts = [100, 100, 100, 100]
        simpson = calculate_simpson_index(counts)
        # Expected: 4 * (100*99) / (400*399) ≈ 0.25
        assert 0.24 < simpson < 0.26

    def test_simpson_index_single_category(self):
        """Test Simpson index when all in one category."""
        counts = [100, 0, 0, 0]
        simpson = calculate_simpson_index(counts)
        # Expected: 100*99 / (100*99) = 1.0
        assert abs(simpson - 1.0) < 0.01

    def test_zipf_coefficient(self):
        """Test Zipf coefficient calculation."""
        # Create Zipf-like distribution
        counts = [1000, 500, 333, 250, 200]
        result = calculate_zipf_coefficient(counts)

        # For perfect Zipf, alpha ≈ 1
        assert result.alpha > 0
        assert 0 <= result.r_squared <= 1


class TestTextAnalyzer:
    """Tests for the main TextAnalyzer class."""

    def test_analyze_simple_text(self):
        """Test analysis of a simple text."""
        analyzer = TextAnalyzer.create()
        result = analyzer.analyze("Мой дядя самых честных правил")

        assert result.n_words == 5
        assert result.shannon_entropy is not None
        assert result.shannon_entropy > 0
        assert result.mean_rank is not None

    def test_analyze_empty_text(self):
        """Test analysis of empty text."""
        analyzer = TextAnalyzer.create()
        result = analyzer.analyze("")

        assert result.n_words == 0
        assert result.shannon_entropy is None

    def test_analyze_with_bootstrap(self):
        """Test analysis with bootstrap confidence interval."""
        analyzer = TextAnalyzer.create()
        text = "Мой дядя самых честных правил когда не в шутку занемог"
        result = analyzer.analyze(text, include_bootstrap=True, bootstrap_iterations=100)

        assert result.bootstrap is not None
        assert result.bootstrap.ci_lower < result.bootstrap.ci_upper

    def test_batch_analysis(self):
        """Test batch analysis of multiple texts."""
        analyzer = TextAnalyzer.create()
        texts = [
            ("text1", "Мой дядя самых честных правил"),
            ("text2", "Я помню чудное мгновенье"),
            ("text3", "Белеет парус одинокий"),
        ]

        batch = analyzer.analyze_batch(texts)

        assert len(batch.results) == 3
        assert batch.extended_stats is not None
        assert batch.correlation is not None

    def test_compare_texts(self):
        """Test text comparison."""
        analyzer = TextAnalyzer.create()
        text1 = "Мой дядя самых честных правил когда не в шутку занемог"
        text2 = "Я помню чудное мгновенье передо мной явилась ты"

        comparison = analyzer.compare(text1, text2, "Пушкин", "Также Пушкин")

        assert comparison.js_divergence >= 0
        assert comparison.cosine_similarity is not None

    def test_rolling_entropy(self):
        """Test rolling entropy analysis."""
        analyzer = TextAnalyzer.create()
        # Need longer text for rolling analysis
        text = " ".join(["мама мыла раму папа пилил доску"] * 20)

        result = analyzer.rolling_entropy(text, window_size=10, step_size=5)

        assert len(result.positions) > 0
        assert len(result.entropies) == len(result.positions)

    def test_dataframe_conversion(self):
        """Test conversion to Polars DataFrame."""
        analyzer = TextAnalyzer.create()
        result = analyzer.analyze("Мой дядя самых честных правил")

        df = analyzer.to_dataframe(result)

        assert len(df) == 29  # rus29 alphabet (33 - ё - й - ъ - ь)
        assert "letter" in df.columns
        assert "probability" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
