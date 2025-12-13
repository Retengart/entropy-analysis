import pytest

from entropy_analysis.core.recognition_batch import (
    Segment,
    train_and_run_batch,
)


def test_train_and_predict_batch():
    segments = [
        Segment(text="aa bb cc", label="a", name="s1"),
        Segment(text="dd ee", label="b", name="s2"),
        Segment(text="ff gg hh", label="a", name="s3"),
        Segment(text="ii jj kk ll", label="b", name="s4"),
    ]

    result = train_and_run_batch(segments, smoothing=1e-3, error_target=0.2)

    assert len(result.tables.class_labels) == 2
    assert abs(sum(result.tables.priors) - 1) < 1e-9
    assert result.recognition.min_error >= 0

    preds = {p.name: p.predicted_label for p in result.predictions}
    assert set(preds.values()).issubset(set(result.tables.class_labels))


def test_requires_labels():
    segments = [Segment(text="no label", name="s1")]
    with pytest.raises(ValueError):
        train_and_run_batch(segments)

