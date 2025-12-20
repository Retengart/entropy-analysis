"""
Batch recognition tab for labeled text segments.
"""

from __future__ import annotations

import json
from typing import List

import streamlit as st
import polars as pl
import plotly.express as px

from entropy_analysis.core.recognition import NoiseConfig as RecognitionNoiseConfig
from entropy_analysis.core.recognition_batch import (
    Segment,
    train_and_run_batch,
)
from entropy_analysis.models.schemas import RecognitionBatchRequest, TextSegment


def _default_segments_json() -> str:
    sample = [
        {
            "name": "Пушкин_1",
            "label": "pushkin",
            "text": "Мой дядя самых честных правил,\nКогда не в шутку занемог...",
        },
        {
            "name": "Лермонтов_1",
            "label": "lermontov",
            "text": "Белеет парус одинокий\nВ тумане моря голубом...",
        },
        {
            "name": "Пушкин_2",
            "label": "pushkin",
            "text": "Люблю грозу в начале мая,\nКогда весенний, первый гром...",
        },
        {
            "name": "Лермонтов_2",
            "label": "lermontov",
            "text": "Выхожу один я на дорогу;\nСквозь туман кремнистый путь блестит...",
        },
    ]
    return json.dumps(sample, ensure_ascii=False, indent=2)


def _render_tables(tables: dict):
    st.subheader("Обученные таблицы")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Классов", len(tables["classes"]))
        st.json({"priors": tables["priors"]})
    with col2:
        st.json({"feature_values": tables["feature_values"]})

    for fname, cond in tables["conditionals"].items():
        df = pl.DataFrame(cond, schema=tables["feature_values"][fname]).with_columns(
            pl.Series("class", tables["classes"])
        )
        df = df.select(pl.col("class"), *tables["feature_values"][fname])
        st.markdown(f"**P(x | A) для признака {fname}**")
        st.dataframe(df, use_container_width=True)


def _render_predictions(preds: list[dict]):
    st.subheader("Предсказания по сегментам")
    rows = []
    for p in preds:
        max_lbl = max(p["posteriors"], key=p["posteriors"].get)
        rows.append(
            {
                "name": p["name"],
                "true_label": p["true_label"],
                "predicted": p["predicted_label"],
                "max_p": p["posteriors"][max_lbl],
            }
        )
    st.dataframe(pl.DataFrame(rows), use_container_width=True)

    # Distribution of predicted classes
    pred_counts = (
        pl.DataFrame(rows)
        .group_by("predicted")
        .count()
        .rename({"count": "n"})
        .sort("n", descending=True)
    )
    if pred_counts.height > 0:
        fig = px.bar(pred_counts.to_pandas(), x="predicted", y="n", title="Распределение предсказанных классов")
        st.plotly_chart(fig, use_container_width=True)


def recognition_batch_tab():
    """Render batch recognition UI."""
    st.header("🧭 Пакетное распознавание")
    st.caption("Обучение таблиц вероятностей по размеченным сегментам и предсказание классов.")

    source = st.radio("Источник сегментов", ["JSON", "Файлы"], horizontal=True)

    delimiter = st.text_input("Разделитель сегментов", value="***", help="Например, ====== или ***")
    uploaded_files = []

    if source == "JSON":
        default_json = _default_segments_json()
        segments_json = st.text_area(
            "Сегменты (JSON, поля: text, label, name)",
            value=st.session_state.get("rec_batch_json", default_json),
            height=260,
        )
    else:
        uploaded_files = st.file_uploader(
            "Загрузите один или несколько файлов (txt/md). Каждый файл — отдельный класс (label = имя файла).",
            type=["txt", "md"],
            accept_multiple_files=True,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        smoothing = st.number_input("Сглаживание", min_value=1e-6, max_value=0.1, value=0.001, step=0.001)
    with col2:
        error_target = st.number_input("Порог ошибки P(e)_zad", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
    with col3:
        noise_factor = st.number_input("Шум (factor)", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
        noise_seed = st.number_input("Seed", min_value=0, max_value=10_000, value=123, step=1)

    col4, col5 = st.columns(2)
    with col4:
        max_features = st.number_input("Топ-K признаков (0 = все)", min_value=0, max_value=50, value=0, step=1)
        max_features_val = None if max_features == 0 else max_features
    with col5:
        min_info = st.number_input("Мин. информативность", min_value=0.0, max_value=1.0, value=0.0, step=0.001)

    profile = st.selectbox(
        "Набор признаков",
        options=[
            ("full", "Полный (все признаки)"),
            ("compact", "Компактный"),
            ("baseline", "Базовый (стабильный)"),
            ("poetry_essential", "🎭 Поэзия (оптимально)"),
            ("poetry_full", "🎭 Поэзия (максимум)"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]
    
    use_lexicons = st.checkbox(
        "📚 Автоматические лексиконы авторов (TF-IDF)",
        value=True,
        help="Извлекает характерные слова для каждого автора. Работает с любыми авторами!"
    )

    if st.button("🔍 Обучить и предсказать", type="primary"):
        segments_input: List[TextSegment] = []

        if source == "JSON":
            try:
                parsed = json.loads(segments_json)
            except json.JSONDecodeError:
                st.error("Некорректный JSON сегментов.")
                return
            try:
                segments_input = [TextSegment.model_validate(item) for item in parsed]
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ошибка валидации: {exc}")
                return
            st.session_state.rec_batch_json = segments_json
        else:
            if not uploaded_files:
                st.error("Загрузите хотя бы один файл.")
                return
            for f in uploaded_files:
                f.seek(0)
                raw = f.read()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("cp1251")
                parts = [p.strip() for p in text.split(delimiter) if p.strip()]
                label = f.name.rsplit(".", 1)[0]
                for idx, part in enumerate(parts, start=1):
                    segments_input.append(
                        TextSegment(text=part, label=label, name=f"{label}_{idx}")
                    )
            if not segments_input:
                st.error("Не удалось найти сегменты по заданному разделителю.")
                return

        req = RecognitionBatchRequest(
            segments=segments_input,
            error_target=error_target,
            smoothing=smoothing,
            noise=(
                {"factor": noise_factor, "mode": "uniform", "seed": noise_seed, "renormalize": True}
                if noise_factor > 0
                else None
            ),
            max_features=max_features_val,
            min_informativeness=min_info,
            feature_profile=profile,
            use_author_lexicons=use_lexicons,
        )

        try:
            noise_cfg = (
                RecognitionNoiseConfig(
                    factor=req.noise.factor,
                    mode=req.noise.mode,
                    seed=req.noise.seed,
                    renormalize=req.noise.renormalize,
                )
                if req.noise
                else None
            )
            result = train_and_run_batch(
                segments=[Segment(text=s.text, name=s.name, label=s.label) for s in req.segments],
                smoothing=req.smoothing,
                noise=noise_cfg,
                error_target=req.error_target,
                max_features=req.max_features,
                min_informativeness=req.min_informativeness,
                feature_profile=req.feature_profile,
                use_author_lexicons=req.use_author_lexicons,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка расчёта: {exc}")
            return

        st.success("Обучение и распознавание завершены.")

        # Recognition run summary
        st.subheader("Метрики признаков")
        
        # Access clean run for features
        clean_run = result.recognition.clean
        
        feat_rows = []
        for f in clean_run.features:
            feat_rows.append(
                {
                    "name": f.name,
                    "informativeness": f.informativeness,
                    "error": f.error,
                    "passes_threshold": f.passes_threshold,
                }
            )
        st.dataframe(pl.DataFrame(feat_rows), use_container_width=True)

        # Show noise effect if available
        if noise_factor > 0:
            st.divider()
            st.markdown(f"### 📉 Влияние помех (factor={noise_factor})")
            
            if result.recognition.delta_rel is not None:
                d_abs = result.recognition.delta_abs
                d_rel = result.recognition.delta_rel
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Ошибка без шума P(e)", f"{result.recognition.clean.min_error:.4f}")
                c2.metric("Ошибка с шумом P*(e)", f"{result.recognition.noisy.min_error:.4f}", delta=f"{d_abs:+.4f}", delta_color="inverse")
                c3.metric("Относ. прирост δP", f"{d_rel:.1%}", delta_color="inverse")

                # Show noisy errors comparison
                if result.recognition.noisy:
                    noisy_map = {f.name: f.error for f in result.recognition.noisy.features}
                    comparison_rows = []
                    for f in clean_run.features:
                        noisy_err = noisy_map.get(f.name, 0.0)
                        diff = noisy_err - f.error
                        # Show only features where error changed significantly
                        if abs(diff) > 1e-5:
                            comparison_rows.append({
                                "Признак": f.name,
                                "P(e) чистый": f.error,
                                "P*(e) шумный": noisy_err,
                                "Разница": diff
                            })
                    
                    if comparison_rows:
                        st.caption("Признаки, наиболее чувствительные к шуму:")
                        # Sort by biggest absolute change
                        comparison_rows.sort(key=lambda x: abs(x["Разница"]), reverse=True)
                        st.dataframe(
                            pl.DataFrame(comparison_rows),
                            use_container_width=True,
                            column_config={
                                "P(e) чистый": st.column_config.NumberColumn(format="%.4f"),
                                "P*(e) шумный": st.column_config.NumberColumn(format="%.4f"),
                                "Разница": st.column_config.NumberColumn(format="%+.4f"),
                            }
                        )
                    else:
                        st.info("Шум не оказал существенного влияния на ошибки отдельных признаков.")
            else:
                st.warning("Результаты с шумом не были вычислены (возможно, ошибка на сервере).")
        
        st.divider()

        # Charts: informativeness and errors
        if feat_rows:
            feat_df = pl.DataFrame(feat_rows)
            # ... existing charts ...
            fig_info = px.bar(
                feat_df.to_pandas(),
                x="name",
                y="informativeness",
                title="Информативность признаков",
            )
            st.plotly_chart(fig_info, use_container_width=True)

            fig_err = px.bar(
                feat_df.to_pandas(),
                x="name",
                y="error",
                title="Ошибки P(e) по признакам (без шума)",
            )
            st.plotly_chart(fig_err, use_container_width=True)

        _render_tables(_tables_to_dict(result))
        _render_predictions([p.__dict__ for p in result.predictions])

        # Conclusion / summary
        true_labels = [p.true_label for p in result.predictions if p.true_label]
        pred_labels = [p.predicted_label for p in result.predictions]
        
        summary_md = []
        
        if true_labels and len(true_labels) == len(result.predictions):
            acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
            avg_conf = sum(p.posteriors[p.predicted_label] for p in result.predictions) / len(result.predictions)
            summary_md.append(f"**Точность (без шума):** `{acc:.3%}`")
            summary_md.append(f"**Средняя уверенность:** `{avg_conf:.3f}`")
            
            # Если есть шумные предсказания, считаем точность для них
            if result.noisy_predictions:
                noisy_pred_labels = [p.predicted_label for p in result.noisy_predictions]
                acc_noisy = sum(t == p for t, p in zip(true_labels, noisy_pred_labels)) / len(true_labels)
                summary_md.append(f"**Точность (с шумом):** `{acc_noisy:.3%}`")
                summary_md.append(f"**Падение точности:** `{(acc - acc_noisy):.3%}`")
            
            if len(set(pred_labels)) == 1:
                summary_md.append("⚠️ Модель склоняется к одному классу.")
            else:
                summary_md.append("✅ Предсказания распределены по классам.")
                
            st.info("\n\n".join(summary_md))
        else:
            avg_conf = sum(p.posteriors[p.predicted_label] for p in result.predictions) / len(result.predictions)
            st.info(
                f"Итог: рассчитаны постериоры по {len(result.predictions)} сегментам, средняя уверенность {avg_conf:.3f}. "
                "Точные метки не заданы, точность не вычислена."
            )

        st.caption(f"Использовано признаков: {len(result.used_features)} ({', '.join(result.used_features)})")


def _tables_to_dict(result):
    conds = {}
    for feat in result.tables.features:
        conds[feat.name] = feat.conditional.tolist()
    return {
        "classes": result.tables.class_labels,
        "priors": result.tables.priors.tolist(),
        "feature_values": result.tables.feature_values,
        "conditionals": conds,
    }

