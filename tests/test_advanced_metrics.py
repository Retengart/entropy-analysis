
import sys
from pathlib import Path

# Add src to path so that imports work
sys.path.append(str(Path(__file__).parent.parent))

from entropy_analysis.core.attribution import (
    calculate_burrows_delta,
    calculate_cosine_delta,
    calculate_eders_delta,
    calculate_z_scores,
)
from entropy_analysis.core.metrics import (
    calculate_gries_dp,
    calculate_hurst_from_word_lengths,
    calculate_mattr,
)
import numpy as np

def test_mattr():
    """Test Moving Average Type-Token Ratio."""
    # Case 1: Short text, small window
    tokens = ["a", "b", "a", "c", "a", "b"]
    # Window size 3
    # Win 1: a,b,a -> TTR = 2/3
    # Win 2: b,a,c -> TTR = 3/3
    # Win 3: a,c,a -> TTR = 2/3
    # Win 4: c,a,b -> TTR = 3/3
    # Mean = (2/3 + 1 + 2/3 + 1) / 4 = (10/3) / 4 = 10/12 = 5/6 ≈ 0.8333
    mattr = calculate_mattr(tokens, window_size=3)
    assert abs(mattr - 5/6) < 1e-6
    
    # Case 2: Empty
    assert calculate_mattr([]) == 0.0
    
    # Case 3: Window > len
    assert calculate_mattr(["a", "b"], window_size=5) == 1.0

def test_gries_dp():
    """Test Gries' Deviation of Proportions."""
    # Perfectly distributed: a b a b
    tokens = ["a", "b", "a", "b"]
    # Segments: [a, b], [a, b]
    # Word 'a': freq=2. Seg 1: 1, Seg 2: 1. Expected: 1, 1. DP = 0.
    res = calculate_gries_dp(tokens, "a", n_segments=2)
    assert res.dp == 0.0
    assert res.dp_norm == 1.0
    
    # Concentrated: a a b b
    tokens = ["a", "a", "b", "b"]
    # Segments: [a, a], [b, b]
    # Word 'a': freq=2. Seg 1: 2, Seg 2: 0. Total size 4.
    # Seg sizes: 2, 2.
    # Obs props: 2/2=1.0, 0/2=0.0.
    # Exp props: 2/4=0.5, 2/4=0.5.
    # DP = 0.5 * (|1-0.5| + |0-0.5|) = 0.5 * (0.5 + 0.5) = 0.5
    # Min seg prop = 0.5.
    # Norm = 1 - 0.5 / (1 - 0.5) = 1 - 1 = 0.
    res = calculate_gries_dp(tokens, "a", n_segments=2)
    assert abs(res.dp - 0.5) < 1e-6
    assert abs(res.dp_norm - 0.0) < 1e-6

def test_hurst():
    """Test Hurst exponent calculation."""
    # Random noise (should be ~0.5)
    np.random.seed(42)
    tokens = ["word"] * 100 # Just for length
    # We mock the internal call or just check structure
    # Actually calculate_hurst_from_word_lengths uses lengths
    # Let's make random lengths
    lengths = np.random.randint(1, 10, 1000)
    tokens = ["a" * l for l in lengths]
    
    res = calculate_hurst_from_word_lengths(tokens)
    # For random noise, H should be close to 0.5
    assert 0.4 < res.hurst_exponent < 0.6
    assert res.interpretation == "random"

def test_delta_metrics():
    """Test Attribution Delta metrics."""
    # Simple case
    # Features: word1, word2
    # Text A: 0.1, 0.9
    # Text B: 0.2, 0.8
    # Means: 0.15, 0.85
    # Stds: 0.05, 0.05 (approx)
    
    f1 = np.array([10, 90], dtype=float)
    f2 = np.array([20, 80], dtype=float)
    # Normalized freqs
    p1 = f1 / 100
    p2 = f2 / 100
    
    all_freqs = np.vstack([p1, p2])
    means = np.mean(all_freqs, axis=0)
    stds = np.std(all_freqs, axis=0, ddof=1)
    
    z1 = calculate_z_scores(p1, means, stds)
    z2 = calculate_z_scores(p2, means, stds)
    
    # Burrows
    # Z1 = [-0.707, 0.707] (approx, since just 2 samples std is tricky)
    # Actually with ddof=1 and 2 samples, std is proportional to difference
    
    delta_b = calculate_burrows_delta(z1, z2)
    assert delta_b > 0
    
    # Cosine
    delta_c = calculate_cosine_delta(z1, z2)
    assert 0 <= delta_c <= 2
    
    # Identical texts -> 0 delta
    assert calculate_burrows_delta(z1, z1) == 0.0
    # Cosine distance of identical vectors is 0
    assert abs(calculate_cosine_delta(z1, z1)) < 1e-9

if __name__ == "__main__":
    test_mattr()
    test_gries_dp()
    test_hurst()
    test_delta_metrics()
    print("✅ All advanced metrics tests passed!")

