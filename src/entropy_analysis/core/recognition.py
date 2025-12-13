"""
Probabilistic recognition algorithm (lab 40) with noise-aware evaluation.

Implements steps 1–11 from the lab:
- per-feature informativeness (eq. 3)
- per-feature error probability (eq. 4)
- joint error for feature pairs (eqs. 5–8)
- effectiveness deltas with/without noise (eq. 1)

Design goals:
- deterministic, pure calculations (no global state)
- strict validation/renormalization of probability tables
- numerical safety (clipping, epsilon guards)
- complexity: O(M · N · R_k) per feature, O(M · R_k · R_q) per pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class NoiseConfig:
    """Noise injection settings."""

    factor: float = 0.0  # additive uniform noise in [-factor, factor]
    mode: str = "uniform"
    seed: int | None = None
    renormalize: bool = True


@dataclass(frozen=True)
class FeatureSpec:
    """Definition of a discrete feature with per-class conditional probabilities."""

    name: str
    values_count: int
    conditional: np.ndarray  # shape (M, values_count)


@dataclass(frozen=True)
class RecognitionInput:
    """Validated recognition input."""

    priors: np.ndarray  # shape (M,)
    features: list[FeatureSpec]
    error_target: float = 0.05
    noise: NoiseConfig | None = None


@dataclass
class FeatureMetrics:
    """Metrics for a single feature."""

    index: int
    name: str
    values_count: int
    informativeness: float
    error: float
    px: list[float]
    passes_threshold: bool


@dataclass
class PairMetrics:
    """Metrics for a pair of features."""

    indices: tuple[int, int]
    names: tuple[str, str]
    error: float
    passes_threshold: bool


@dataclass
class RecognitionRun:
    """Analysis results for one run (clean or noisy)."""

    features: list[FeatureMetrics] = field(default_factory=list)
    best_feature: FeatureMetrics | None = None
    best_pair: PairMetrics | None = None
    min_error: float = 1.0


@dataclass
class RecognitionAnalysis:
    """Full analysis including clean and optional noisy evaluation."""

    clean: RecognitionRun
    noisy: RecognitionRun | None
    delta_abs: float | None
    delta_rel: float | None


class RecognitionEngine:
    """Engine implementing the probabilistic recognition algorithm."""

    @staticmethod
    def sample_input() -> RecognitionInput:
        """Return deterministic sample matching the lab example (M=3, N=2, R=[2,3])."""
        priors = np.array([0.3, 0.5, 0.2], dtype=float)
        feature1 = FeatureSpec(
            name="feature_1",
            values_count=2,
            conditional=np.array(
                [
                    [0.7, 0.3],  # class 1
                    [0.4, 0.6],  # class 2
                    [0.5, 0.5],  # class 3
                ],
                dtype=float,
            ),
        )
        feature2 = FeatureSpec(
            name="feature_2",
            values_count=3,
            conditional=np.array(
                [
                    [0.2, 0.5, 0.3],  # class 1
                    [0.3, 0.4, 0.3],  # class 2
                    [0.1, 0.6, 0.3],  # class 3
                ],
                dtype=float,
            ),
        )
        return RecognitionInput(priors=priors, features=[feature1, feature2], error_target=0.05)

    @classmethod
    def analyze(cls, raw_input: RecognitionInput) -> RecognitionAnalysis:
        """Validate, run clean analysis, optionally run noisy analysis, and compute deltas."""
        validated = cls._validate_input(raw_input)
        clean_run = cls._run(validated)
        noisy_run: RecognitionRun | None = None
        delta_abs: float | None = None
        delta_rel: float | None = None

        if validated.noise and validated.noise.factor > 0:
            noisy_input = cls._apply_noise(validated)
            noisy_run = cls._run(noisy_input)
            delta_abs, delta_rel = cls._delta(clean_run.min_error, noisy_run.min_error)

        return RecognitionAnalysis(clean=clean_run, noisy=noisy_run, delta_abs=delta_abs, delta_rel=delta_rel)

    @staticmethod
    def _validate_input(raw: RecognitionInput) -> RecognitionInput:
        """Ensure priors and conditionals are well-formed and normalized."""
        priors = np.asarray(raw.priors, dtype=float)
        if priors.ndim != 1 or priors.size == 0:
            raise ValueError("P(A_i) must be a 1D vector with at least one class.")
        if not np.all(np.isfinite(priors)) or np.any(priors < 0):
            raise ValueError("P(A_i) must be finite and non-negative.")
        priors_sum = float(priors.sum())
        if priors_sum <= 0:
            raise ValueError("P(A_i) sum must be > 0.")
        priors /= priors_sum

        if not 0 < raw.error_target < 1:
            raise ValueError("error_target must be in (0, 1).")

        if not raw.features:
            raise ValueError("At least one feature must be provided.")

        features: list[FeatureSpec] = []
        for idx, feat in enumerate(raw.features):
            if feat.values_count <= 1:
                raise ValueError(f"Feature {feat.name} must have at least 2 values.")

            cond = np.asarray(feat.conditional, dtype=float)
            if cond.shape != (priors.size, feat.values_count):
                raise ValueError(
                    f"Feature {feat.name} has shape {cond.shape}, expected {(priors.size, feat.values_count)}."
                )
            if not np.all(np.isfinite(cond)) or np.any(cond < 0):
                raise ValueError(f"Feature {feat.name} has non-finite or negative probabilities.")

            normed = []
            for cls_idx, row in enumerate(cond):
                row_sum = float(row.sum())
                if row_sum <= 0:
                    raise ValueError(
                        f"Feature {feat.name}, class {cls_idx} has zero total probability."
                    )
                normed.append(row / row_sum)
            normed_cond = np.stack(normed, axis=0)
            features.append(
                FeatureSpec(
                    name=feat.name or f"feature_{idx+1}",
                    values_count=feat.values_count,
                    conditional=normed_cond,
                )
            )

        noise = raw.noise
        if noise:
            if noise.factor < 0:
                raise ValueError("noise.factor must be non-negative.")
            if noise.mode not in ("uniform",):
                raise ValueError("noise.mode must be 'uniform'.")

        return RecognitionInput(
            priors=priors, features=features, error_target=raw.error_target, noise=noise
        )

    @staticmethod
    def _apply_noise(validated: RecognitionInput) -> RecognitionInput:
        """Inject additive noise into conditional probabilities and renormalize per class."""
        assert validated.noise is not None  # for type-checkers
        cfg = validated.noise
        rng = np.random.default_rng(cfg.seed)

        noisy_features: list[FeatureSpec] = []
        for feat in validated.features:
            noise = rng.uniform(-cfg.factor, cfg.factor, size=feat.conditional.shape)
            perturbed = feat.conditional + noise
            perturbed = np.clip(perturbed, 0.0, None)

            if cfg.renormalize:
                renorm_rows = []
                for cls_idx, row in enumerate(perturbed):
                    row_sum = float(row.sum())
                    if row_sum <= EPS:
                        # Fallback to uniform to avoid degeneracy
                        renorm_rows.append(np.full_like(row, 1.0 / row.size))
                    else:
                        renorm_rows.append(row / row_sum)
                perturbed = np.stack(renorm_rows, axis=0)

            noisy_features.append(
                FeatureSpec(name=feat.name, values_count=feat.values_count, conditional=perturbed)
            )

        return RecognitionInput(
            priors=validated.priors,
            features=noisy_features,
            error_target=validated.error_target,
            noise=validated.noise,
        )

    @classmethod
    def _run(cls, data: RecognitionInput) -> RecognitionRun:
        """Run clean or noisy analysis."""
        features_metrics = []
        for idx, feat in enumerate(data.features):
            info = cls._informativeness(data.priors, feat.conditional)
            error, px = cls._feature_error(data.priors, feat.conditional)
            features_metrics.append(
                FeatureMetrics(
                    index=idx,
                    name=feat.name,
                    values_count=feat.values_count,
                    informativeness=info,
                    error=error,
                    px=px,
                    passes_threshold=error < data.error_target,
                )
            )

        # Sort by informativeness descending
        features_metrics.sort(key=lambda f: f.informativeness, reverse=True)

        best_feature = min(features_metrics, key=lambda f: f.error, default=None)

        best_pair = cls._best_pair(data.priors, data.features, data.error_target)

        candidate_errors: list[float] = []
        if best_feature:
            candidate_errors.append(best_feature.error)
        if best_pair:
            candidate_errors.append(best_pair.error)
        if not candidate_errors:
            candidate_errors.append(1.0)

        min_error = min(candidate_errors)

        return RecognitionRun(
            features=features_metrics,
            best_feature=best_feature,
            best_pair=best_pair,
            min_error=min_error,
        )

    @staticmethod
    def _feature_error(priors: np.ndarray, conditional: np.ndarray) -> tuple[float, list[float]]:
        """Compute P(e)_k (eq. 4) and P(x_l^k) for each value."""
        px: list[float] = []
        error = 0.0
        for val_idx in range(conditional.shape[1]):
            class_terms = priors * conditional[:, val_idx]
            p_x = float(class_terms.sum())
            px.append(p_x)
            if p_x <= EPS:
                continue
            error += p_x - float(class_terms.max())
        return error, px

    @staticmethod
    def _informativeness(priors: np.ndarray, conditional: np.ndarray) -> float:
        """Compute Shannon informativeness (eq. 3)."""
        h_a = -float(np.sum(priors * np.log2(np.clip(priors, EPS, None))))
        accum = 0.0
        for val_idx in range(conditional.shape[1]):
            joint = priors * conditional[:, val_idx]
            p_x = float(joint.sum())
            if p_x <= EPS:
                continue
            posterior = joint / p_x
            accum += p_x * float(np.sum(posterior * np.log2(np.clip(posterior, EPS, None))))
        return h_a + accum

    @classmethod
    def _best_pair(cls, priors: np.ndarray, features: Sequence[FeatureSpec], error_target: float) -> PairMetrics | None:
        """Evaluate all feature pairs and return the best (minimum error)."""
        if len(features) < 2:
            return None

        best_error = None
        best_pair: PairMetrics | None = None

        for i, j in combinations(range(len(features)), 2):
            err = cls._pair_error(priors, features[i].conditional, features[j].conditional)
            if best_error is None or err < best_error:
                best_error = err
                best_pair = PairMetrics(
                    indices=(i, j),
                    names=(features[i].name, features[j].name),
                    error=err,
                    passes_threshold=err < error_target,
                )
        return best_pair

    @staticmethod
    def _pair_error(priors: np.ndarray, cond_a: np.ndarray, cond_b: np.ndarray) -> float:
        """Compute P(e) for two features (eqs. 5–8)."""
        error = 0.0
        r_a = cond_a.shape[1]
        r_b = cond_b.shape[1]
        for va, vb in product(range(r_a), range(r_b)):
            cond_product = cond_a[:, va] * cond_b[:, vb]
            p_bj = float(np.sum(priors * cond_product))
            if p_bj <= EPS:
                continue
            error += p_bj - float(np.max(priors * cond_product))
        return error

    @staticmethod
    def _delta(clean_error: float, noisy_error: float) -> tuple[float, float]:
        """Compute absolute and relative error delta (eq. 1)."""
        delta_abs = noisy_error - clean_error
        if noisy_error <= EPS:
            return delta_abs, 0.0
        delta_rel = delta_abs / noisy_error
        return delta_abs, delta_rel


def build_recognition_input(
    priors: Iterable[float],
    features: Sequence[dict],
    error_target: float = 0.05,
    noise: NoiseConfig | None = None,
) -> RecognitionInput:
    """
    Helper to construct RecognitionInput from plain python structures (used by API/UI).
    """
    feature_specs: list[FeatureSpec] = []
    for idx, feat in enumerate(features):
        name = feat.get("name") or f"feature_{idx+1}"
        values_count = int(feat["values_count"])
        conditional = np.asarray(feat["conditional"], dtype=float)
        feature_specs.append(
            FeatureSpec(name=name, values_count=values_count, conditional=conditional)
        )
    priors_arr = np.asarray(list(priors), dtype=float)
    return RecognitionInput(priors=priors_arr, features=feature_specs, error_target=error_target, noise=noise)
