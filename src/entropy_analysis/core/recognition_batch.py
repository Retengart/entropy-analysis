"""
Batch recognition utilities for text segments.

This module:
- extracts discrete features from text,
- trains priors P(A_i) and conditional tables P(x|A_i),
- predicts posteriors for new segments,
- produces RecognitionInput for the core RecognitionEngine.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from entropy_analysis.core.recognition import (
    FeatureSpec,
    NoiseConfig,
    RecognitionInput,
    RecognitionRun,
    RecognitionEngine,
)

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


# ---------- Feature definitions ----------


@dataclass(frozen=True)
class DiscreteFeature:
    name: str
    values: List[str]
    extractor: Callable[[str], str]

    def index_of(self, value: str) -> int:
        try:
            return self.values.index(value)
        except ValueError:
            raise ValueError(f"Unexpected value '{value}' for feature '{self.name}'") from None


def bin_average_word_length(text: str) -> str:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return "empty"
    avg = sum(len(t) for t in tokens) / len(tokens)
    if avg <= 4:
        return "short"
    if avg <= 7:
        return "medium"
    if avg <= 12:
        return "long"
    return "very_long"


def bin_line_length(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    avg = sum(len(ln) for ln in lines) / len(lines)
    if avg <= 30:
        return "short_line"
    if avg <= 60:
        return "medium_line"
    if avg <= 90:
        return "long_line"
    return "very_long_line"


def bin_punctuation_ratio(text: str) -> str:
    if not text:
        return "empty"
    punct_chars = sum(1 for ch in text if ch in ".,;:!?—-–()[]{}\"'«»")
    ratio = punct_chars / max(len(text), 1)
    if ratio < 0.05:
        return "low_punct"
    if ratio < 0.1:
        return "mid_punct"
    if ratio < 0.2:
        return "high_punct"
    return "very_high_punct"


BUILTIN_FEATURES: List[DiscreteFeature] = [
    DiscreteFeature(
        name="avg_word_len",
        values=["empty", "short", "medium", "long", "very_long"],
        extractor=bin_average_word_length,
    ),
    DiscreteFeature(
        name="avg_line_len",
        values=["empty", "short_line", "medium_line", "long_line", "very_long_line"],
        extractor=bin_line_length,
    ),
    DiscreteFeature(
        name="punct_ratio",
        values=["empty", "low_punct", "mid_punct", "high_punct", "very_high_punct"],
        extractor=bin_punctuation_ratio,
    ),
]


# ---------- Data structures ----------


@dataclass
class Segment:
    text: str
    name: str | None = None
    label: str | None = None


@dataclass
class TrainedTables:
    class_labels: List[str]
    features: List[FeatureSpec]
    feature_values: Dict[str, List[str]]
    priors: np.ndarray  # shape (M,)


@dataclass
class SegmentPrediction:
    name: str | None
    true_label: str | None
    predicted_label: str
    posteriors: Dict[str, float]
    feature_values: Dict[str, str]


@dataclass
class RecognitionBatchResult:
    tables: TrainedTables
    recognition: RecognitionRun
    predictions: List[SegmentPrediction]


# ---------- Core helpers ----------


def _compute_feature_values(
    text: str, features: Sequence[DiscreteFeature]
) -> Dict[str, str]:
    return {feat.name: feat.extractor(text) for feat in features}


def _train_tables(
    segments: Sequence[Segment],
    features: Sequence[DiscreteFeature],
    smoothing: float,
) -> TrainedTables:
    labeled = [s for s in segments if s.label]
    if not labeled:
        raise ValueError("Для обучения таблиц нужно указать метки классов (label) у сегментов.")

    class_labels = sorted({s.label for s in labeled if s.label is not None})
    m = len(class_labels)
    if m < 2:
        raise ValueError("Нужно минимум два класса для распознавания.")

    class_index = {lbl: idx for idx, lbl in enumerate(class_labels)}
    priors = np.zeros(m, dtype=float)

    # counts per feature per class per value
    feature_value_map: Dict[str, List[str]] = {f.name: list(f.values) for f in features}
    counts: Dict[str, np.ndarray] = {}
    for feat in features:
        counts[feat.name] = np.zeros((m, len(feat.values)), dtype=float)

    for seg in labeled:
        cls_idx = class_index[seg.label]  # label not None due to filtered
        priors[cls_idx] += 1.0
        feat_vals = _compute_feature_values(seg.text, features)
        for feat in features:
            val = feat_vals[feat.name]
            val_idx = feat.index_of(val)
            counts[feat.name][cls_idx, val_idx] += 1.0

    if priors.sum() <= 0:
        raise ValueError("Не удалось посчитать априорные вероятности.")
    priors /= priors.sum()

    feature_specs: List[FeatureSpec] = []
    for feat in features:
        raw = counts[feat.name] + smoothing
        # normalize each row
        normed_rows = []
        for row in raw:
            s = float(row.sum())
            if s <= 0:
                normed_rows.append(np.full_like(row, 1.0 / len(row)))
            else:
                normed_rows.append(row / s)
        conditional = np.stack(normed_rows, axis=0)
        feature_specs.append(
            FeatureSpec(
                name=feat.name,
                values_count=len(feat.values),
                conditional=conditional,
            )
        )

    return TrainedTables(
        class_labels=class_labels,
        features=feature_specs,
        feature_values=feature_value_map,
        priors=priors,
    )


def _predict_segments(
    tables: TrainedTables,
    features: Sequence[DiscreteFeature],
    segments: Sequence[Segment],
) -> List[SegmentPrediction]:
    preds: List[SegmentPrediction] = []

    # Map feature name -> discrete values for index lookup
    feature_value_lists = {f.name: f.values for f in features}

    for seg in segments:
        vals = _compute_feature_values(seg.text, features)
        log_probs = np.log(np.clip(tables.priors, 1e-12, None))
        for feat_spec in tables.features:
            val = vals[feat_spec.name]
            try:
                v_idx = feature_value_lists[feat_spec.name].index(val)
            except ValueError:
                # unseen value -> use minimal probability guard
                v_idx = None
            cond_row = feat_spec.conditional
            if v_idx is None:
                # fallback: uniform minimal penalty
                log_probs += math.log(1e-9)
            else:
                log_probs += np.log(np.clip(cond_row[:, v_idx], 1e-12, None))

        # convert to posterior
        max_log = float(np.max(log_probs))
        stabilized = np.exp(log_probs - max_log)
        posterior = stabilized / stabilized.sum()
        pred_idx = int(np.argmax(posterior))
        pred_label = tables.class_labels[pred_idx]
        preds.append(
            SegmentPrediction(
                name=seg.name,
                true_label=seg.label,
                predicted_label=pred_label,
                posteriors={lbl: float(posterior[i]) for i, lbl in enumerate(tables.class_labels)},
                feature_values=vals,
            )
        )
    return preds


def train_and_run_batch(
    segments: Sequence[Segment],
    smoothing: float = 1e-3,
    noise: NoiseConfig | None = None,
    error_target: float = 0.05,
    features: Sequence[DiscreteFeature] | None = None,
) -> RecognitionBatchResult:
    """
    Train probability tables from labeled segments and run recognition analysis + predictions.
    """
    feats = list(features) if features else BUILTIN_FEATURES
    tables = _train_tables(segments, feats, smoothing=smoothing)
    rec_input = RecognitionInput(
        priors=tables.priors,
        features=tables.features,
        error_target=error_target,
        noise=noise,
    )
    rec_analysis = RecognitionEngine.analyze(rec_input)
    preds = _predict_segments(tables, feats, segments)
    return RecognitionBatchResult(
        tables=tables,
        recognition=rec_analysis.clean,
        predictions=preds,
    )

