"""
Recognition algorithm (lab 40) tab.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

from entropy_analysis.core.recognition import (
    NoiseConfig as RecognitionNoiseConfig,
    RecognitionAnalysis,
    RecognitionEngine,
    RecognitionInput,
    RecognitionRun,
    build_recognition_input,
)
from entropy_analysis.models.schemas import RecognitionRequest


def _default_spec() -> str:
    sample = RecognitionEngine.sample_input()
    payload = {
        "use_sample": True,
        "classes": ["A1", "A2", "A3"],
        "priors": sample.priors.tolist(),
        "features": [
            {
                "name": f.name,
                "values_count": f.values_count,
                "conditional": f.conditional.tolist(),
            }
            for f in sample.features
        ],
        "error_target": sample.error_target,
        "noise": {"factor": 0.0, "mode": "uniform", "seed": 123, "renormalize": True},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _to_input(request: RecognitionRequest) -> RecognitionInput:
    noise_cfg = (
        RecognitionNoiseConfig(
            factor=request.noise.factor,
            mode=request.noise.mode,
            seed=request.noise.seed,
            renormalize=request.noise.renormalize,
        )
        if request.noise
        else None
    )

    if request.use_sample:
        sample = RecognitionEngine.sample_input()
        return RecognitionInput(
            priors=sample.priors,
            features=sample.features,
            error_target=request.error_target,
            noise=noise_cfg,
        )

    if len(request.classes) != len(request.priors):
        raise ValueError("Длины classes и priors должны совпадать.")

    feature_dicts: List[Dict[str, Any]] = []
    for feat in request.features:
        if len(feat.conditional) != len(request.priors):
            raise ValueError(
                f"Feature {feat.name or '<unnamed>'}: число строк conditional "
                "должно совпадать с числом классов."
            )
        feature_dicts.append(
            {
                "name": feat.name,
                "values_count": feat.values_count,
                "conditional": feat.conditional,
            }
        )

    return build_recognition_input(
        priors=request.priors,
        features=feature_dicts,
        error_target=request.error_target,
        noise=noise_cfg,
    )


def _chart_bar(title: str, names: list[str], values: list[float]) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color="#2E86AB",
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(title=title, xaxis_title="Признак", yaxis_title="")
    return fig


def _render_run(run: RecognitionRun, title: str):
    st.subheader(title)

    cols = st.columns(2)
    if run.best_feature:
        cols[0].metric(
            "Лучший признак",
            f"{run.best_feature.name}",
            help=f"P(e)={run.best_feature.error:.3f}",
        )
    if run.best_pair:
        pair_names = " + ".join(run.best_pair.names)
        cols[1].metric(
            "Лучшая пара",
            pair_names,
            help=f"P(e)={run.best_pair.error:.3f}",
        )

    table = [
        {
            "index": f.index,
            "name": f.name,
            "values": f.values_count,
            "informativeness": round(f.informativeness, 6),
            "error": round(f.error, 6),
            "passes_threshold": f.passes_threshold,
        }
        for f in run.features
    ]
    st.dataframe(table, use_container_width=True)

    names = [f.name for f in run.features]
    infos = [f.informativeness for f in run.features]
    errors = [f.error for f in run.features]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_chart_bar("Информативность", names, infos), use_container_width=True)
    with c2:
        st.plotly_chart(_chart_bar("Вероятность ошибки P(e)", names, errors), use_container_width=True)


def recognition_tab(options):
    """Render the recognition tab."""
    st.header("🧭 Алгоритм распознавания образов")
    st.caption("Вероятностный классификатор по признакам с оценкой ошибок и влияния помех.")

    default_spec = _default_spec()
    spec_text = st.text_area(
        "Описание задачи (JSON, см. пример ниже)",
        value=st.session_state.get("recognition_spec", default_spec),
        height=320,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Использовать встроенный пример"):
            spec_text = default_spec
            st.session_state.recognition_spec = default_spec
            st.rerun()
    with col2:
        if st.button("Очистить ввод"):
            spec_text = default_spec
            st.session_state.recognition_spec = default_spec
            st.rerun()

    if st.button("🔍 Рассчитать", type="primary"):
        try:
            payload = json.loads(spec_text)
        except json.JSONDecodeError as exc:  # noqa: F841
            st.error("Некорректный JSON. Проверьте синтаксис.")
            return

        try:
            request = RecognitionRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка валидации входных данных: {exc}")
            return

        try:
            input_data = _to_input(request)
            analysis: RecognitionAnalysis = RecognitionEngine.analyze(input_data)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка расчета: {exc}")
            return

        st.session_state.recognition_spec = spec_text
        st.success("Расчет завершен.")

        _render_run(analysis.clean, "Без помех")

        if analysis.noisy:
            st.divider()
            st.info(
                f"ΔP = {analysis.delta_abs:.4f}, "
                f"δP = {analysis.delta_rel:.4f} (по формуле 1)",
                icon="🔎",
            )
            _render_run(analysis.noisy, "С помехами")

