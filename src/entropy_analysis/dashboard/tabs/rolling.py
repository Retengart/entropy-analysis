"""
Rolling analysis and Split analysis tabs.
"""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from entropy_analysis.dashboard.components.metrics import render_full_report
from entropy_analysis.visualization.charts import (
    create_correlation_scatter,
    create_entropy_histogram,
    create_rolling_entropy_chart,
)


def split_analysis(options):
    """Split text analysis tab - analyze text segments separated by delimiter."""
    st.header("✂️ Анализ с разделителями")

    st.markdown("""
    Этот режим позволяет анализировать несколько текстов, разделённых специальным маркером.
    Программа автоматически разделит текст и построит график зависимости **энтропии от количества слов**
    с линией линейной регрессии.
    """)

    # Input method
    input_method = st.radio(
        "Способ ввода",
        ["Вставить текст", "Загрузить файл"],
        horizontal=True,
        key="split_input_method",
    )

    text = ""

    if input_method == "Вставить текст":
        text = st.text_area(
            "Введите текст с разделителями",
            height=300,
            placeholder="Текст 1\n\n***\n\nТекст 2\n\n***\n\nТекст 3",
            key="split_text_input",
        )
    else:
        uploaded_file = st.file_uploader(
            "Выберите файл с текстами", type=["txt", "md"], key="split_file_upload"
        )
        if uploaded_file:
            uploaded_file.seek(0)
            raw_content = uploaded_file.read()
            try:
                text = raw_content.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_content.decode("cp1251")
            st.text_area("Содержимое файла", text, height=200, disabled=True)

    # Delimiter configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        delimiter = st.text_input(
            "Разделитель сегментов",
            value="***",
            help="Символы или строка, разделяющие сегменты текста",
        )
    with col2:
        st.metric("Разделитель", f'"{delimiter}"')

    # Show example
    with st.expander("📖 Пример использования"):
        st.code(
            """Мой дядя самых честных правил,
Когда не в шутку занемог

***

Белеет парус одинокий
В тумане моря голубом!

***

Ночь, улица, фонарь, аптека,
Бессмысленный и тусклый свет.""",
            language="text",
        )
        st.caption("Каждый сегмент будет проанализирован отдельно, затем построен график H vs N")

    if st.button("🔍 Анализировать сегменты", type="primary", key="split_analyze_btn"):
        if not text.strip():
            st.warning("Пожалуйста, введите текст для анализа")
            return

        status_container = st.empty()
        
        # Функция для логирования
        def log_callback(msg: str):
            if options.get("show_verbose_logs"):
                # Check if message is a progress update [X/Y]
                if msg.startswith("[") and "]" in msg and "/" in msg:
                    # Update status label instead of appending new line
                    status.update(label=f"Анализ сегментов: {msg}")
                else:
                    status.write(f"⏳ {msg}")

        # Используем st.status для логов если включено
        if options.get("show_verbose_logs"):
            with st.status("Анализ сегментов...", expanded=True) as status:
                batch_result = st.session_state.analyzer.split_and_analyze(
                    text=text,
                    delimiter=delimiter,
                    auto_name=True,
                    include_bootstrap=options["include_bootstrap"],
                    include_advanced_metrics=options["include_advanced_metrics"],
                    log_callback=log_callback,
                )
                status.update(label="Анализ завершен!", state="complete", expanded=False)
        else:
            with st.spinner("Разделяем и анализируем сегменты..."):
                batch_result = st.session_state.analyzer.split_and_analyze(
                    text=text,
                    delimiter=delimiter,
                    auto_name=True,
                    include_bootstrap=options["include_bootstrap"],
                    include_advanced_metrics=options["include_advanced_metrics"],
                )

        n_segments = len(batch_result.results)

        if n_segments == 0:
            st.error(f"❌ Не найдено сегментов с разделителем '{delimiter}'")
            st.info("Попробуйте изменить разделитель или проверьте формат текста")
            return

        st.success(f"✅ Найдено и проанализировано сегментов: **{n_segments}**")

        # Statistics section
        st.subheader("📊 Статистика энтропии")

        if batch_result.extended_stats:
            stats = batch_result.extended_stats
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Среднее H", f"{stats.mean:.4f} бит")
            with col2:
                st.metric("Медиана H", f"{stats.median:.4f} бит")
            with col3:
                st.metric("Стд. откл.", f"{stats.std_dev:.4f}")
            with col4:
                st.metric("Диапазон", f"[{stats.min_value:.4f}, {stats.max_value:.4f}]")

        # Correlation section
        if batch_result.correlation is not None:
            st.subheader("📈 Корреляция: Энтропия (H) vs Количество слов (N)")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Коэффициент корреляции (r)",
                    f"{batch_result.correlation:.4f}",
                    help="Сила линейной связи между H и N",
                )
            with col2:
                st.metric(
                    "Коэффициент детерминации (R²)",
                    f"{batch_result.correlation_r_squared:.4f}",
                    help="Доля дисперсии H, объясняемая N",
                )
            with col3:
                st.metric(
                    "p-value",
                    f"{batch_result.correlation_p_value:.6f}",
                    help="Статистическая значимость корреляции",
                )

            # Regression equation
            if batch_result.correlation_slope is not None:
                st.info(
                    f"**Уравнение линейной регрессии:**  \n"
                    f"H = **{batch_result.correlation_slope:.6f}** × N + **{batch_result.correlation_intercept:.4f}**"
                )

            # Interpretation
            st.subheader("💡 Интерпретация")

            # Strength of correlation
            r = abs(batch_result.correlation)
            if r > 0.7:
                strength = "**сильная**"
                strength_color = "🟢"
            elif r > 0.4:
                strength = "**умеренная**"
                strength_color = "🟡"
            else:
                strength = "**слабая**"
                strength_color = "🔴"

            direction = "положительная" if batch_result.correlation > 0 else "отрицательная"

            st.write(
                f"{strength_color} Обнаружена {strength} {direction} корреляция (r = {batch_result.correlation:.4f})"
            )

            # Statistical significance
            if batch_result.correlation_p_value < 0.05:
                st.success("✓ Корреляция статистически значима (p < 0.05)")
            else:
                st.warning("⚠ Корреляция статистически незначима (p ≥ 0.05)")

        # Visualizations
        st.subheader("📊 Визуализации")

        tab1, tab2, tab3, tab4 = st.tabs(["📈 График H vs N", "📊 Гистограмма", "📋 Таблица", "📑 ВСЁ"])

        with tab1:
            st.markdown("**График зависимости энтропии от количества слов с линией регрессии**")
            fig = create_correlation_scatter(
                batch_result,
                show_trendline=True,
                highlight_outliers=True,
                label_top_n=min(5, n_segments),
            )
            # Update title
            fig.update_layout(
                title=f"Энтропия vs Количество слов<br><sub>{n_segments} сегментов, разделитель: '{delimiter}'</sub>"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"split_tab1_scatter_{id(batch_result)}")

        with tab2:
            st.markdown("**Распределение значений энтропии**")
            fig = create_entropy_histogram(batch_result)
            st.plotly_chart(fig, use_container_width=True, key=f"split_tab2_hist_{id(batch_result)}")

        with tab3:
            st.markdown("**Таблица результатов по сегментам**")

            # Create detailed table
            df_data = []
            for i, (name, result) in enumerate(batch_result.results, 1):
                df_data.append(
                    {
                        "#": i,
                        "Сегмент": name,
                        "Слов (N)": result.n_words,
                        "Энтропия (H)": f"{result.shannon_entropy:.4f}"
                        if result.shannon_entropy
                        else "—",
                        "Средний ранг (x̄)": f"{result.mean_rank:.2f}" if result.mean_rank else "—",
                        "σ (ранг)": f"{result.std_rank:.2f}" if result.std_rank else "—",
                        "Уник. букв": result.n_unique_letters,
                    }
                )

            df_display = pl.DataFrame(df_data)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Download options
            col1, col2 = st.columns(2)

            with col1:
                # Full dataframe with all metrics
                df_full = st.session_state.analyzer.batch_to_dataframe(batch_result)
                csv = df_full.write_csv()
                st.download_button(
                    "📥 Скачать полную таблицу (CSV)",
                    csv,
                    f"split_analysis_{n_segments}_segments.csv",
                    "text/csv",
                )

            with col2:
                # JSON with correlation data
                import json

                json_data = {
                    "delimiter": delimiter,
                    "n_segments": n_segments,
                    "correlation": {
                        "r": batch_result.correlation,
                        "r_squared": batch_result.correlation_r_squared,
                        "p_value": batch_result.correlation_p_value,
                        "slope": batch_result.correlation_slope,
                        "intercept": batch_result.correlation_intercept,
                    },
                    "segments": [
                        {
                            "name": name,
                            "n_words": r.n_words,
                            "entropy": r.shannon_entropy,
                            "mean_rank": r.mean_rank,
                            "std_rank": r.std_rank,
                        }
                        for name, r in batch_result.results
                    ],
                }
                json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 Скачать данные (JSON)",
                    json_str,
                    f"split_analysis_{n_segments}_segments.json",
                    "application/json",
                )
                
        with tab4:
            st.header("Полный отчет по сегментам")
            st.markdown(f"Всего сегментов: **{len(batch_result.results)}**")
            
            # Show full report for first 5 and last 5 segments if too many
            results_to_show = batch_result.results
            if len(results_to_show) > 10:
                st.info("Показаны первые 5 и последние 5 сегментов.")
                results_to_show = results_to_show[:5] + results_to_show[-5:]
                
            for name, res in results_to_show:
                with st.expander(f"📄 {name}"):
                    render_full_report(res)


def rolling_analysis(options):
    """Rolling entropy analysis tab."""
    st.header("📈 Скользящая энтропия")

    text = st.text_area("Введите текст", height=200, key="rolling_text")

    col1, col2 = st.columns(2)
    with col1:
        window_size = st.slider("Размер окна (слова)", 10, 200, 50)
    with col2:
        step_size = st.slider("Шаг", 1, 50, 10)

    if st.button("🔍 Анализировать динамику", type="primary"):
        if not text.strip():
            st.warning("Введите текст для анализа")
            return

        with st.spinner("Анализируем динамику энтропии..."):
            result = st.session_state.analyzer.rolling_entropy(
                text, window_size=window_size, step_size=step_size
            )

        if not result.positions:
            st.warning("Текст слишком короткий для заданного размера окна")
            return

        st.success("✅ Анализ завершён!")

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Среднее H", f"{result.mean_entropy:.4f}")
        with col2:
            st.metric("Стд. откл.", f"{result.std_entropy:.4f}")
        with col3:
            st.metric("Мин H", f"{result.min_entropy:.4f}")
        with col4:
            st.metric("Макс H", f"{result.max_entropy:.4f}")

        # Chart
        fig = create_rolling_entropy_chart(result)
        st.plotly_chart(fig, use_container_width=True, key=f"rolling_chart_{id(result)}")

