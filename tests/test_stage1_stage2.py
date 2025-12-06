"""
Tests for Stage 1 & 2 improvements:
- All letters counting (not just first letters)
- Stable entropy calculations with xlogy
- LexicalRichness integration
- N-gram entropy improvements
"""

import numpy as np
import pytest

from entropy_analysis.core.analyze import TextAnalyzer
from entropy_analysis.core.metrics import (
    calculate_kl_divergence,
    calculate_lexical_richness_metrics,
    calculate_ngram_distribution,
    calculate_ngram_entropy,
    _entropy_from_counts,
)
from entropy_analysis.core.normalize import create_normalizer


class TestAllLettersCounting:
    """Test that we count ALL letters, not just first letters."""

    def test_count_letters_vs_first_letters(self):
        """Verify count_letters counts all letters, not just first."""
        normalizer = create_normalizer()
        text = "мама мыла раму"
        
        # First letters: м, м, р (3 letters)
        first_counts = normalizer.count_first_letters(text)
        assert sum(first_counts.values()) == 3
        
        # All letters: м, а, м, а, м, ы, л, а, р, а, м, у (12 letters)
        all_counts = normalizer.count_letters(text)
        assert sum(all_counts.values()) == 12
        
        # Verify specific counts
        assert all_counts.get("м", 0) == 4  # appears 4 times total
        assert all_counts.get("а", 0) == 4
        assert first_counts.get("м", 0) == 2  # only 2 words start with м

    def test_analyzer_uses_all_letters(self):
        """Verify TextAnalyzer uses all letters for entropy calculation."""
        analyzer = TextAnalyzer.create()
        text = "мама мыла раму папа пилил доску"
        
        result = analyzer.analyze(text)
        
        # Should have many letters (not just 6 first letters)
        assert result.n_words == 6
        # n_letters should be much larger than n_words
        # Each word has multiple letters
        assert result.shannon_entropy is not None
        assert result.shannon_entropy > 0
        
        # Verify letter_stats uses correct probability base
        total_prob = sum(stat.probability for stat in result.letter_stats)
        # Should sum to ~1.0 (allowing for floating point errors)
        assert abs(total_prob - 1.0) < 0.01


class TestStableEntropy:
    """Test stable entropy calculations using xlogy."""

    def test_entropy_from_counts_handles_zeros(self):
        """Test that _entropy_from_counts handles zero counts safely."""
        # Uniform distribution
        counts = np.array([10, 10, 10, 10], dtype=np.int64)
        h1 = _entropy_from_counts(counts)
        assert abs(h1 - 2.0) < 1e-10  # log2(4) = 2
        
        # With zeros
        counts_with_zeros = np.array([10, 10, 0, 0], dtype=np.int64)
        h2 = _entropy_from_counts(counts_with_zeros)
        assert h2 > 0
        assert h2 < h1  # Less entropy with fewer categories
        
        # All zeros
        counts_all_zeros = np.array([0, 0, 0, 0], dtype=np.int64)
        h3 = _entropy_from_counts(counts_all_zeros)
        assert h3 == 0.0

    def test_kl_divergence_uses_xlogy(self):
        """Test that KL divergence uses xlogy for numerical stability."""
        # Test with distributions that have zeros
        p = np.array([0.5, 0.5, 0.0, 0.0])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        
        result = calculate_kl_divergence(p, q)
        
        # Should compute without errors
        assert result.divergence >= 0
        assert np.isfinite(result.divergence)
        
        # Test edge case: p has zeros where q has values
        p2 = np.array([1.0, 0.0, 0.0])
        q2 = np.array([0.33, 0.33, 0.34])
        result2 = calculate_kl_divergence(p2, q2)
        # Should handle gracefully (epsilon smoothing)
        assert np.isfinite(result2.divergence)

    def test_ngram_distribution_uses_stable_entropy(self):
        """Test that n-gram distribution uses stable entropy calculation."""
        tokens = list("мамамылараму")
        
        result = calculate_ngram_distribution(tokens, n=2, top_k=10)
        
        assert result is not None
        assert result.entropy > 0
        assert np.isfinite(result.entropy)
        assert result.conditional_entropy >= 0
        assert np.isfinite(result.conditional_entropy)


class TestNgramEntropy:
    """Test improved n-gram entropy calculations."""

    def test_ngram_entropy_bigrams(self):
        """Test bigram entropy calculation."""
        tokens = list("мамамылараму")
        
        h = calculate_ngram_entropy(tokens, n=2)
        
        assert h >= 0
        assert np.isfinite(h)
        # For a text with some structure, entropy should be reasonable
        assert h < 10.0  # Upper bound sanity check

    def test_ngram_entropy_trigrams(self):
        """Test trigram entropy calculation."""
        tokens = list("мамамыларамупапапилилдоску")
        
        h = calculate_ngram_entropy(tokens, n=3)
        
        assert h >= 0
        assert np.isfinite(h)
        # Trigram entropy should be <= bigram entropy (more context)
        h_bigram = calculate_ngram_entropy(tokens, n=2)
        assert h <= h_bigram + 0.1  # Allow small numerical differences

    def test_ngram_entropy_short_text(self):
        """Test n-gram entropy with insufficient data."""
        tokens = list("ма")
        
        h = calculate_ngram_entropy(tokens, n=2)
        assert h == 0.0  # Not enough tokens


class TestLexicalRichness:
    """Test LexicalRichness integration."""

    def test_lexical_richness_metrics_basic(self):
        """Test basic lexical richness metrics calculation."""
        tokens = ["мама", "мыла", "раму", "папа", "пилил", "доску"] * 10
        
        metrics = calculate_lexical_richness_metrics(tokens)
        
        # Should return dict with expected keys
        assert isinstance(metrics, dict)
        assert "yules_k" in metrics
        assert "mtld" in metrics
        assert "mattr" in metrics
        assert "hdd" in metrics
        
        # Values should be reasonable
        if metrics["yules_k"] is not None:
            assert metrics["yules_k"] >= 0
        if metrics["mtld"] is not None:
            assert metrics["mtld"] > 0
        if metrics["mattr"] is not None:
            assert 0 <= metrics["mattr"] <= 1
        if metrics["hdd"] is not None:
            assert 0 <= metrics["hdd"] <= 1

    def test_lexical_richness_empty_tokens(self):
        """Test lexical richness with empty token list."""
        metrics = calculate_lexical_richness_metrics([])
        
        assert metrics["yules_k"] is None or metrics["yules_k"] == 0
        assert metrics["mtld"] is None or metrics["mtld"] == 0
        assert metrics["mattr"] is None or metrics["mattr"] == 0

    def test_lexical_richness_diverse_text(self):
        """Test lexical richness with diverse vocabulary."""
        # More diverse text
        tokens = ["мама", "мыла", "раму", "папа", "пилил", "доску", 
                  "кот", "собака", "птица", "рыба", "дерево", "цветок"] * 5
        
        metrics = calculate_lexical_richness_metrics(tokens)
        
        # More diverse text should have reasonable MATTR
        if metrics["mattr"] is not None:
            assert metrics["mattr"] > 0.1  # Reasonable lower bound (adjusted for short text)

    def test_analyzer_includes_lexical_metrics(self):
        """Test that TextAnalyzer includes lexical richness metrics."""
        analyzer = TextAnalyzer.create()
        text = "мама мыла раму папа пилил доску кот собака птица рыба"
        
        result = analyzer.analyze(text)
        
        # Should have lexical metrics
        assert result.yules_k is not None or result.yules_k == 0
        assert result.mtld is not None
        assert result.mattr is not None
        assert result.hdd is not None  # New metric from LexicalRichness


class TestIntegration:
    """Integration tests for Stage 1 & 2 changes."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline with all improvements."""
        analyzer = TextAnalyzer.create()
        text = """
        Мой дядя самых честных правил,
        Когда не в шутку занемог,
        Он уважать себя заставил
        И лучше выдумать не мог.
        """
        
        result = analyzer.analyze(text, include_advanced_metrics=True)
        
        # Basic metrics
        assert result.n_words > 0
        assert result.shannon_entropy is not None
        assert result.shannon_entropy > 0
        
        # Letter-level n-grams
        assert result.letter_bigram_entropy is not None
        assert result.letter_trigram_entropy is not None
        
        # Lexical metrics
        assert result.mtld is not None
        assert result.mattr is not None
        assert result.hdd is not None
        
        # Verify probabilities sum correctly (using n_letters, not n_words)
        total_letter_prob = sum(
            stat.probability for stat in result.letter_stats
        )
        assert abs(total_letter_prob - 1.0) < 0.01

    def test_compare_uses_correct_counts(self):
        """Test that comparison uses correct letter counting."""
        analyzer = TextAnalyzer.create()
        text1 = "мама мыла раму"
        text2 = "папа пилил доску"
        
        comparison = analyzer.compare(text1, text2)
        
        # Should compute divergence without errors
        assert comparison.js_divergence >= 0
        assert np.isfinite(comparison.js_divergence)
        assert comparison.kl_divergence_p_q >= 0
        assert np.isfinite(comparison.kl_divergence_p_q)

    def test_batch_analysis_consistency(self):
        """Test that batch analysis is consistent with single analysis."""
        analyzer = TextAnalyzer.create()
        text = "мама мыла раму папа пилил доску"
        
        single_result = analyzer.analyze(text)
        batch_result = analyzer.analyze_batch([("test", text)])
        
        assert len(batch_result.results) == 1
        batch_single = batch_result.results[0][1]
        
        # Entropy should match
        assert abs(single_result.shannon_entropy - batch_single.shannon_entropy) < 1e-10
        # Word count should match
        assert single_result.n_words == batch_single.n_words


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

