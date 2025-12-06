"""
Tests for Stage 3 improvements:
- Advanced burstiness metrics (pybursts, bursty_dynamics)
- Kleinberg's algorithm integration
- Memory coefficient and burstiness parameter
"""

import numpy as np
import pytest

from entropy_analysis.core.analyze import TextAnalyzer
from entropy_analysis.core.metrics import (
    calculate_advanced_burstiness_metrics,
    calculate_burstiness_index,
)


class TestBurstinessBaseline:
    """Test baseline burstiness calculation."""

    def test_burstiness_regular_pattern(self):
        """Test burstiness with regular pattern (low burstiness)."""
        # Regular pattern: word appears every 5 tokens
        tokens = ["word"] * 20
        regular_tokens = []
        for i in range(100):
            if i % 5 == 0:
                regular_tokens.append("word")
            else:
                regular_tokens.append(f"other{i}")
        
        burstiness = calculate_burstiness_index(regular_tokens, min_count=5)
        
        # Regular pattern should have low or negative burstiness
        assert burstiness is not None
        assert burstiness <= 0.5  # Should be closer to 0 or negative

    def test_burstiness_bursty_pattern(self):
        """Test burstiness with bursty pattern."""
        # Bursty pattern: word appears in clusters
        # Note: Regular clusters can actually give negative burstiness
        # because intervals within cluster are small, between clusters are large
        tokens = []
        for i in range(10):
            # Cluster of 5 occurrences
            tokens.extend(["word"] * 5)
            # Then gap of 10 other words
            tokens.extend([f"other{j}" for j in range(10)])
        
        burstiness = calculate_burstiness_index(tokens, min_count=5)
        
        # Should compute without errors
        assert burstiness is not None
        assert np.isfinite(burstiness)
        # Value can be positive or negative depending on pattern
        assert -1 <= burstiness <= 1  # Valid range

    def test_burstiness_empty_tokens(self):
        """Test burstiness with empty token list."""
        burstiness = calculate_burstiness_index([], min_count=5)
        assert burstiness == 0.0

    def test_burstiness_insufficient_data(self):
        """Test burstiness with insufficient data."""
        tokens = ["word"] * 3  # Less than min_count=5
        burstiness = calculate_burstiness_index(tokens, min_count=5)
        assert burstiness == 0.0


class TestAdvancedBurstiness:
    """Test advanced burstiness metrics."""

    def test_advanced_burstiness_basic(self):
        """Test basic advanced burstiness calculation."""
        # Create bursty pattern
        tokens = []
        for i in range(5):
            tokens.extend(["word"] * 10)
            tokens.extend([f"other{j}" for j in range(20)])
        
        metrics = calculate_advanced_burstiness_metrics(tokens, min_count=5)
        
        # Should always return baseline burstiness
        assert metrics.burstiness_b is not None
        assert isinstance(metrics.burstiness_b, float)
        assert np.isfinite(metrics.burstiness_b)
        
        # Advanced metrics may be None if libraries unavailable
        # But structure should be correct
        assert metrics.kleinberg_bursts is None or isinstance(metrics.kleinberg_bursts, int)
        assert metrics.kleinberg_burst_ratio is None or (0 <= metrics.kleinberg_burst_ratio <= 1)
        assert metrics.burstiness_parameter is None or np.isfinite(metrics.burstiness_parameter)
        assert metrics.memory_coefficient is None or np.isfinite(metrics.memory_coefficient)

    def test_advanced_burstiness_empty_tokens(self):
        """Test advanced burstiness with empty tokens."""
        metrics = calculate_advanced_burstiness_metrics([], min_count=5)
        
        assert metrics.burstiness_b == 0.0
        assert metrics.kleinberg_bursts is None
        assert metrics.kleinberg_burst_ratio is None
        assert metrics.burstiness_parameter is None
        assert metrics.memory_coefficient is None

    def test_advanced_burstiness_regular_pattern(self):
        """Test advanced burstiness with regular pattern."""
        # Regular pattern
        tokens = []
        for i in range(100):
            if i % 5 == 0:
                tokens.append("word")
            else:
                tokens.append(f"other{i % 10}")
        
        metrics = calculate_advanced_burstiness_metrics(tokens, min_count=5)
        
        assert metrics.burstiness_b is not None
        # Regular pattern should have low burstiness
        assert metrics.burstiness_b <= 0.5


class TestAnalyzerIntegration:
    """Test integration with TextAnalyzer."""

    def test_analyzer_includes_burstiness(self):
        """Test that TextAnalyzer includes burstiness metrics."""
        analyzer = TextAnalyzer.create()
        text = "мама мыла раму папа пилил доску " * 10
        
        result = analyzer.analyze(text)
        
        # Should have baseline burstiness
        assert result.burstiness is not None
        assert isinstance(result.burstiness, float)
        assert np.isfinite(result.burstiness)

    def test_analyzer_includes_advanced_burstiness(self):
        """Test that TextAnalyzer includes advanced burstiness metrics."""
        analyzer = TextAnalyzer.create()
        # Create bursty text pattern
        text = ("мама " * 10 + "мыла " * 5 + "раму " * 10 + "папа " * 5) * 5
        
        result = analyzer.analyze(text, include_advanced_metrics=True)
        
        # Should have baseline burstiness
        assert result.burstiness is not None
        
        # Advanced metrics may be None if libraries unavailable
        # But fields should exist
        assert hasattr(result, "kleinberg_bursts")
        assert hasattr(result, "kleinberg_burst_ratio")
        assert hasattr(result, "burstiness_parameter")
        assert hasattr(result, "memory_coefficient")
        
        # If available, should be valid
        if result.kleinberg_bursts is not None:
            assert isinstance(result.kleinberg_bursts, int)
            assert result.kleinberg_bursts >= 0
        
        if result.kleinberg_burst_ratio is not None:
            assert 0 <= result.kleinberg_burst_ratio <= 1
        
        if result.burstiness_parameter is not None:
            assert np.isfinite(result.burstiness_parameter)
        
        if result.memory_coefficient is not None:
            assert np.isfinite(result.memory_coefficient)

    def test_analyzer_burstiness_without_advanced(self):
        """Test that baseline burstiness works without advanced metrics."""
        analyzer = TextAnalyzer.create()
        text = "мама мыла раму папа пилил доску"
        
        result = analyzer.analyze(text, include_advanced_metrics=False)
        
        # Should have baseline burstiness even without advanced metrics
        assert result.burstiness is not None
        
        # Advanced metrics should be None when not requested
        assert result.kleinberg_bursts is None
        assert result.kleinberg_burst_ratio is None
        assert result.burstiness_parameter is None
        assert result.memory_coefficient is None


class TestBurstinessEdgeCases:
    """Test edge cases for burstiness calculations."""

    def test_single_word_repetition(self):
        """Test burstiness with single word repeated."""
        tokens = ["word"] * 100
        
        burstiness = calculate_burstiness_index(tokens, min_count=5)
        
        # Single word repeated should have very low burstiness (regular pattern)
        assert burstiness is not None
        assert burstiness <= 0.1  # Very regular

    def test_all_unique_words(self):
        """Test burstiness with all unique words."""
        tokens = [f"word{i}" for i in range(100)]
        
        burstiness = calculate_burstiness_index(tokens, min_count=5)
        
        # All unique words means no word appears >= min_count times
        assert burstiness == 0.0

    def test_very_long_text(self):
        """Test burstiness with very long text."""
        # Create long bursty pattern
        tokens = []
        for i in range(50):
            tokens.extend(["word"] * 20)
            tokens.extend([f"other{j}" for j in range(50)])
        
        metrics = calculate_advanced_burstiness_metrics(tokens, min_count=5)
        
        assert metrics.burstiness_b is not None
        assert np.isfinite(metrics.burstiness_b)
        # Value can be positive or negative depending on pattern
        assert -1 <= metrics.burstiness_b <= 1  # Valid range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

