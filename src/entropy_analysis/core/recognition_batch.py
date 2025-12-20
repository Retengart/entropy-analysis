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


def bin_unique_ratio(text: str) -> str:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return "empty"
    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < 0.3:
        return "low_unique"
    if unique_ratio < 0.5:
        return "mid_unique"
    if unique_ratio < 0.7:
        return "high_unique"
    return "very_high_unique"


def bin_vowel_ratio(text: str) -> str:
    vowels = set("аеёиоуыэюяaeiou")
    letters = [ch.lower() for ch in text if ch.isalpha()]
    if not letters:
        return "empty"
    ratio = sum(ch in vowels for ch in letters) / len(letters)
    if ratio < 0.35:
        return "low_vowel"
    if ratio < 0.5:
        return "mid_vowel"
    if ratio < 0.65:
        return "high_vowel"
    return "very_high_vowel"


def bin_line_count(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    if n == 0:
        return "empty"
    if n == 1:
        return "one_line"
    if n <= 4:
        return "few_lines"
    if n <= 8:
        return "several_lines"
    return "many_lines"


def bin_token_count(text: str) -> str:
    tokens = WORD_RE.findall(text.lower())
    n = len(tokens)
    if n == 0:
        return "empty"
    if n <= 30:
        return "tiny"
    if n <= 80:
        return "short_text"
    if n <= 150:
        return "medium_text"
    return "long_text"


def bin_long_word_share(text: str) -> str:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return "empty"
    share = sum(len(t) >= 8 for t in tokens) / len(tokens)
    if share < 0.1:
        return "low_long_words"
    if share < 0.25:
        return "mid_long_words"
    if share < 0.4:
        return "high_long_words"
    return "very_high_long_words"


STOP_WORDS = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "она",
    "как",
    "я",
    "с",
    "со",
    "а",
    "то",
    "все",
    "всё",
    "к",
    "на",
    "ты",
    "мы",
    "они",
    "бы",
    "для",
    "же",
    "ли",
    "или",
    "но",
    "о",
    "от",
    "по",
    "из",
    "у",
    "над",
    "под",
    "при",
    "про",
    "это",
    "этот",
    "эта",
    "эти",
}


def bin_stopword_ratio(text: str) -> str:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return "empty"
    sw = sum(t in STOP_WORDS for t in tokens)
    ratio = sw / len(tokens)
    if ratio < 0.15:
        return "low_stop"
    if ratio < 0.3:
        return "mid_stop"
    if ratio < 0.45:
        return "high_stop"
    return "very_high_stop"


def bin_upper_ratio(text: str) -> str:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "empty"
    upper = sum(ch.isupper() for ch in letters) / len(letters)
    if upper < 0.02:
        return "low_upper"
    if upper < 0.08:
        return "mid_upper"
    if upper < 0.15:
        return "high_upper"
    return "very_high_upper"


def bin_sentence_length(text: str) -> str:
    # грубое разбиение по . ! ?; без учета многоточия
    import re as _re

    sentences = [_re.sub(r"\s+", " ", s).strip() for s in _re.split(r"[.!?]", text) if s.strip()]
    if not sentences:
        return "empty"
    lengths = [len(WORD_RE.findall(s.lower())) for s in sentences]
    avg = sum(lengths) / len(lengths) if lengths else 0
    if avg <= 6:
        return "very_short_sent"
    if avg <= 12:
        return "short_sent"
    if avg <= 20:
        return "medium_sent"
    return "long_sent"


RARE_LETTERS = set("щчфъёй")


def bin_rare_letter_ratio(text: str) -> str:
    letters = [ch.lower() for ch in text if ch.isalpha()]
    if not letters:
        return "empty"
    ratio = sum(ch in RARE_LETTERS for ch in letters) / len(letters)
    if ratio < 0.01:
        return "very_low_rare"
    if ratio < 0.025:
        return "low_rare"
    if ratio < 0.05:
        return "mid_rare"
    if ratio < 0.08:
        return "high_rare"
    return "very_high_rare"


def bin_end_punct(text: str) -> str:
    trimmed = text.rstrip()
    if not trimmed:
        return "empty"
    last = trimmed[-1]
    if last == "?":
        return "question_end"
    if last == "!":
        return "exclam_end"
    if last == ";":
        return "semicolon_end"
    if last == ",":
        return "comma_end"
    if last == ".":
        return "dot_end"
    return "other_end"


VOWELS_RU = set("аеёиоуыэюя")


def _count_syllables(word: str) -> int:
    return sum(ch in VOWELS_RU for ch in word.lower())


def bin_syllables_per_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    syllables = []
    for ln in lines:
        words = WORD_RE.findall(ln.lower())
        if not words:
            continue
        syllables.append(sum(_count_syllables(w) for w in words))
    if not syllables:
        return "empty"
    avg = sum(syllables) / len(syllables)
    if avg < 8:
        return "low_syll_per_line"
    if avg < 12:
        return "mid_syll_per_line"
    if avg < 16:
        return "high_syll_per_line"
    return "very_high_syll_per_line"


def bin_line_length_std(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    lens = [len(ln) for ln in lines]
    if len(lens) == 1:
        std = 0.0
    else:
        m = sum(lens) / len(lens)
        std = (sum((x - m) ** 2 for x in lens) / len(lens)) ** 0.5
    if std < 5:
        return "low_line_std"
    if std < 12:
        return "mid_line_std"
    if std < 20:
        return "high_line_std"
    return "very_high_line_std"


def bin_rhyme_repetition(text: str) -> str:
    """
    Approximate rhyme repetition: most frequent 3-letter suffix of line endings.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    suffixes: Dict[str, int] = {}
    for ln in lines:
        words = WORD_RE.findall(ln.lower())
        if not words:
            continue
        last = words[-1]
        suff = last[-3:] if len(last) >= 3 else last
        suffixes[suff] = suffixes.get(suff, 0) + 1
    if not suffixes:
        return "empty"
    top = max(suffixes.values())
    ratio = top / max(len(lines), 1)
    if ratio < 0.15:
        return "very_low_rhyme_rep"
    if ratio < 0.3:
        return "low_rhyme_rep"
    if ratio < 0.5:
        return "mid_rhyme_rep"
    if ratio < 0.7:
        return "high_rhyme_rep"
    return "very_high_rhyme_rep"


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
    DiscreteFeature(
        name="unique_ratio",
        values=["empty", "low_unique", "mid_unique", "high_unique", "very_high_unique"],
        extractor=bin_unique_ratio,
    ),
    DiscreteFeature(
        name="vowel_ratio",
        values=["empty", "low_vowel", "mid_vowel", "high_vowel", "very_high_vowel"],
        extractor=bin_vowel_ratio,
    ),
    DiscreteFeature(
        name="line_count",
        values=["empty", "one_line", "few_lines", "several_lines", "many_lines"],
        extractor=bin_line_count,
    ),
    DiscreteFeature(
        name="token_count",
        values=["empty", "tiny", "short_text", "medium_text", "long_text"],
        extractor=bin_token_count,
    ),
    DiscreteFeature(
        name="long_word_share",
        values=["empty", "low_long_words", "mid_long_words", "high_long_words", "very_high_long_words"],
        extractor=bin_long_word_share,
    ),
    DiscreteFeature(
        name="stopword_ratio",
        values=["empty", "low_stop", "mid_stop", "high_stop", "very_high_stop"],
        extractor=bin_stopword_ratio,
    ),
    DiscreteFeature(
        name="upper_ratio",
        values=["empty", "low_upper", "mid_upper", "high_upper", "very_high_upper"],
        extractor=bin_upper_ratio,
    ),
    DiscreteFeature(
        name="sentence_length",
        values=["empty", "very_short_sent", "short_sent", "medium_sent", "long_sent"],
        extractor=bin_sentence_length,
    ),
    DiscreteFeature(
        name="rare_letter_ratio",
        values=["empty", "very_low_rare", "low_rare", "mid_rare", "high_rare", "very_high_rare"],
        extractor=bin_rare_letter_ratio,
    ),
    DiscreteFeature(
        name="end_punct",
        values=["empty", "question_end", "exclam_end", "semicolon_end", "comma_end", "dot_end", "other_end"],
        extractor=bin_end_punct,
    ),
    DiscreteFeature(
        name="syllables_per_line",
        values=[
            "empty",
            "low_syll_per_line",
            "mid_syll_per_line",
            "high_syll_per_line",
            "very_high_syll_per_line",
        ],
        extractor=bin_syllables_per_line,
    ),
    DiscreteFeature(
        name="line_length_std",
        values=["empty", "low_line_std", "mid_line_std", "high_line_std", "very_high_line_std"],
        extractor=bin_line_length_std,
    ),
    DiscreteFeature(
        name="rhyme_repetition",
        values=[
            "empty",
            "very_low_rhyme_rep",
            "low_rhyme_rep",
            "mid_rhyme_rep",
            "high_rhyme_rep",
            "very_high_rhyme_rep",
        ],
        extractor=bin_rhyme_repetition,
    ),
]

# Try to load poetry-specific features
try:
    from .poetry_features import POETRY_FEATURE_DEFINITIONS
    
    for feat_def in POETRY_FEATURE_DEFINITIONS:
        BUILTIN_FEATURES.append(
            DiscreteFeature(
                name=feat_def["name"],
                values=feat_def["values"],
                extractor=feat_def["extractor"],
            )
        )
except ImportError:
    # Poetry features not available
    pass

# Profiles of feature names
BASELINE_FEATURE_NAMES = {
    "avg_word_len",
    "avg_line_len",
    "punct_ratio",
    "token_count",
    "long_word_share",
    "stopword_ratio",
}

COMPACT_FEATURE_NAMES = {
    "avg_word_len",
    "avg_line_len",
    "punct_ratio",
    "token_count",
    "long_word_share",
    "stopword_ratio",
    "end_punct",
    "line_count",
    "unique_ratio",
}

POETRY_ESSENTIAL_NAMES = {
    # Только проверенные эффективные признаки
    "adjective_density",
    "enjambment_rate",
    "exclamation_ratio",
    "meter_pattern",
    "rhyme_scheme",
    "syllables_per_line",
    "avg_line_len",
    # Лексиконы добавляются автоматически через use_author_lexicons
}

POETRY_FULL_NAMES = {
    # All poetry features + structural baseline
    "meter_pattern",
    "meter_regularity",
    "rhyme_scheme",
    "rhyme_quality",
    "enjambment_rate",
    "adjective_density",
    "pushkin_lexicon",
    "lermontov_lexicon",
    "consonant_repetition",
    "voiced_consonant_ratio",
    "exclamation_ratio",
    "question_ratio",
    "avg_line_len",
    "syllables_per_line",
    "line_length_std",
    "rhyme_repetition",
    "avg_word_len",
    "punct_ratio",
    "stopword_ratio",
}


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
    used_features: List[str]


# ---------- Core helpers ----------


def _compute_feature_values(
    text: str, features: Sequence[DiscreteFeature]
) -> Dict[str, str]:
    return {feat.name: feat.extractor(text) for feat in features}


def _train_tables(
    segments: Sequence[Segment],
    features: Sequence[DiscreteFeature],
    smoothing: float,
    balance_classes: bool = True,
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
    
    # УЛУЧШЕНИЕ: балансировка классов
    # При несбалансированных данных используем равномерные приоры
    if balance_classes:
        print(f"\n⚖️  Балансировка классов:")
        print(f"   Исходные приоры: {dict(zip(class_labels, priors))}")
        priors = np.ones(m) / m  # равномерное распределение
        print(f"   Сбалансированные: {dict(zip(class_labels, priors))}")
    else:
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
    priors: np.ndarray,
    class_labels: Sequence[str],
    feature_specs: Sequence[FeatureSpec],
    features: Sequence[DiscreteFeature],
    segments: Sequence[Segment],
) -> List[SegmentPrediction]:
    preds: List[SegmentPrediction] = []

    # Map feature name -> discrete values for index lookup
    feature_value_lists = {f.name: f.values for f in features}

    for seg in segments:
        vals = _compute_feature_values(seg.text, features)
        log_probs = np.log(np.clip(priors, 1e-12, None))
        for feat_spec in feature_specs:
            val = vals[feat_spec.name]
            try:
                v_idx = feature_value_lists[feat_spec.name].index(val)
            except ValueError:
                # unseen value -> use minimal probability guard
                v_idx = None
            cond_row = feat_spec.conditional
            if v_idx is None:
                # unseen value: assume uniform across classes to avoid arbitrary penalty
                log_probs += math.log(1.0 / len(priors))
            else:
                log_probs += np.log(np.clip(cond_row[:, v_idx], 1e-12, None))

        # convert to posterior
        max_log = float(np.max(log_probs))
        stabilized = np.exp(log_probs - max_log)
        posterior = stabilized / stabilized.sum()
        pred_idx = int(np.argmax(posterior))
        pred_label = class_labels[pred_idx]
        preds.append(
            SegmentPrediction(
                name=seg.name,
                true_label=seg.label,
                predicted_label=pred_label,
                posteriors={lbl: float(posterior[i]) for i, lbl in enumerate(class_labels)},
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
    max_features: int | None = None,
    min_informativeness: float = 0.0,
    feature_profile: str = "full",
    use_author_lexicons: bool = True,
) -> RecognitionBatchResult:
    """
    Train probability tables from labeled segments and run recognition analysis + predictions.
    
    Args:
        use_author_lexicons: если True, автоматически извлекает характерные слова
                             для каждого автора через TF-IDF и добавляет как признаки
    """
    if features:
        feats = list(features)
    else:
        if feature_profile == "baseline":
            feats = [f for f in BUILTIN_FEATURES if f.name in BASELINE_FEATURE_NAMES]
        elif feature_profile == "compact":
            feats = [f for f in BUILTIN_FEATURES if f.name in COMPACT_FEATURE_NAMES]
        elif feature_profile == "poetry_essential":
            feats = [f for f in BUILTIN_FEATURES if f.name in POETRY_ESSENTIAL_NAMES]
        elif feature_profile == "poetry_full":
            feats = [f for f in BUILTIN_FEATURES if f.name in POETRY_FULL_NAMES]
        else:
            feats = BUILTIN_FEATURES
    
    # Автоматическое построение лексиконов для каждого автора (если включено)
    if use_author_lexicons and "poetry" in feature_profile:
        try:
            from .poetry_features import build_author_lexicons, bin_author_lexicon_density
            
            # Строим TF-IDF лексиконы
            build_author_lexicons(segments, top_n=50)
            
            # Добавляем динамические признаки для каждого автора
            authors = sorted({s.label for s in segments if s.label})
            for author in authors:
                # ИСПРАВЛЕНИЕ: используем default argument для правильного замыкания
                def make_extractor(author_name=author):
                    return lambda text: bin_author_lexicon_density(text, author_name)
                
                feats.append(
                    DiscreteFeature(
                        name=f"lexicon_{author}",
                        values=["empty", "unknown_lexicon", "very_low", "low", "mid", "high", "very_high"],
                        extractor=make_extractor(),
                    )
                )
        except ImportError:
            pass  # poetry_features не доступен или sklearn отсутствует
    
    tables = _train_tables(segments, feats, smoothing=smoothing, balance_classes=True)
    full_input = RecognitionInput(
        priors=tables.priors,
        features=tables.features,
        error_target=error_target,
        noise=noise,
    )
    full_analysis = RecognitionEngine.analyze(full_input)

    # Select features by informativeness / threshold
    selected = list(full_analysis.clean.features)
    
    # УЛУЧШЕНИЕ: автоматически отбрасываем признаки с нулевой информативностью
    # (они не помогают различать классы)
    selected = [f for f in selected if f.informativeness > 1e-6]
    
    if min_informativeness > 0:
        selected = [f for f in selected if f.informativeness >= min_informativeness]
    if max_features is not None:
        selected = sorted(selected, key=lambda f: f.informativeness, reverse=True)[:max_features]
    if not selected:
        # fallback: take top-3 most informative (даже если низкие)
        selected = sorted(
            full_analysis.clean.features, 
            key=lambda f: f.informativeness, 
            reverse=True
        )[:min(3, len(full_analysis.clean.features))]

    selected_names = {f.name for f in selected}
    selected_specs = [feat for feat in tables.features if feat.name in selected_names]

    rec_input = RecognitionInput(
        priors=tables.priors,
        features=selected_specs,
        error_target=error_target,
        noise=noise,
    )
    rec_analysis = RecognitionEngine.analyze(rec_input)

    # Map DiscreteFeature list to selected only
    feat_lookup = {f.name: f for f in feats}
    selected_discrete = [feat_lookup[name] for name in selected_names if name in feat_lookup]

    preds = _predict_segments(
        priors=tables.priors,
        class_labels=tables.class_labels,
        feature_specs=selected_specs,
        features=selected_discrete,
        segments=segments,
    )
    return RecognitionBatchResult(
        tables=tables,
        recognition=rec_analysis.clean,
        predictions=preds,
        used_features=[f.name for f in rec_analysis.clean.features],
    )
