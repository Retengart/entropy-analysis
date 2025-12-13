import pytest

from entropy_analysis.core.recognition import (
    NoiseConfig,
    RecognitionEngine,
    RecognitionInput,
    build_recognition_input,
)


def test_sample_metrics_match_lab_example():
    analysis = RecognitionEngine.analyze(RecognitionEngine.sample_input())

    assert analysis.noisy is None
    assert analysis.clean.min_error == pytest.approx(0.472, rel=1e-3)

    f1 = next(f for f in analysis.clean.features if f.name == "feature_1")
    f2 = next(f for f in analysis.clean.features if f.name == "feature_2")

    assert f1.error == pytest.approx(0.49, rel=1e-3)
    assert f2.error == pytest.approx(0.50, rel=1e-3)
    assert f1.informativeness > f2.informativeness
    assert analysis.clean.best_pair is not None
    assert analysis.clean.best_pair.error <= f1.error


def test_noise_is_deterministic_with_seed():
    sample = RecognitionEngine.sample_input()
    noisy_input = RecognitionInput(
        priors=sample.priors,
        features=sample.features,
        error_target=sample.error_target,
        noise=NoiseConfig(factor=0.05, seed=123),
    )

    run1 = RecognitionEngine.analyze(noisy_input)
    run2 = RecognitionEngine.analyze(noisy_input)

    assert run1.noisy is not None
    assert run2.noisy is not None
    assert run1.noisy.min_error == pytest.approx(run2.noisy.min_error, rel=1e-9)
    assert run1.delta_abs == pytest.approx(run2.delta_abs, rel=1e-9)
    assert run1.delta_rel == pytest.approx(run2.delta_rel, rel=1e-9)


def test_validation_rejects_mismatched_shapes():
    priors = [0.6, 0.4]
    features = [
        {"name": "bad", "values_count": 2, "conditional": [[0.5, 0.5], [0.3, 0.7]]},
        # wrong number of classes in conditional (1 row instead of 2)
        {"name": "oops", "values_count": 2, "conditional": [[0.2, 0.8]]},
    ]

    with pytest.raises(ValueError):
        data = build_recognition_input(priors, features)
        RecognitionEngine.analyze(data)

