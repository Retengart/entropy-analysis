"""
Single text analysis tab.
"""

import streamlit as st

from entropy_analysis.dashboard.components.metrics import (
    render_advanced_metrics,
    render_bootstrap_info,
    render_full_report,
    render_metrics,
)
from entropy_analysis.visualization.charts import (
    create_letter_distribution_chart,
    create_zipf_plot,
)


def single_text_analysis(options):
    """Single text analysis tab."""
    st.header("📄 Анализ одного текста")

    # Text input methods
    input_method = st.radio("Способ ввода", ["Вставить текст", "Загрузить файл"], horizontal=True)

    text = ""
    source_name = None

    if input_method == "Вставить текст":
        text = st.text_area(
            "Введите текст для анализа", height=200, placeholder="Вставьте текст здесь..."
        )
        source_name = st.text_input("Название (опционально)")
    else:
        uploaded_file = st.file_uploader("Выберите файл", type=["txt", "md"])
        if uploaded_file:
            uploaded_file.seek(0)  # Reset file position
            raw_content = uploaded_file.read()
            try:
                text = raw_content.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_content.decode("cp1251")
            source_name = uploaded_file.name
            st.text_area("Содержимое файла", text, height=150, disabled=True)

    if st.button("🔍 Анализировать", type="primary"):
        if not text.strip():
            st.warning("Пожалуйста, введите текст для анализа")
            return

        status_container = st.empty()
        
        # Функция для логирования
        def log_callback(msg: str):
            if options.get("show_verbose_logs"):
                with status_container.container():
                    st.text(f"⏳ {msg}")

        # Используем st.status для красивого отображения процесса (только если включены логи)
        if options.get("show_verbose_logs"):
            with st.status("Выполнение анализа...", expanded=True) as status:
                def status_logger(msg: str):
                    status.write(msg)
                
                result = st.session_state.analyzer.analyze(
                    text=text,
                    source_name=source_name,
                    include_bootstrap=options["include_bootstrap"],
                    bootstrap_iterations=options["bootstrap_n"],
                    include_complexity=options["include_complexity"],
                    include_advanced_metrics=options["include_advanced_metrics"],
                    log_callback=status_logger,
                )
                status.update(label="Анализ завершен!", state="complete", expanded=False)
        else:
            with st.spinner("Анализируем текст..."):
                result = st.session_state.analyzer.analyze(
                    text=text,
                    source_name=source_name,
                    include_bootstrap=options["include_bootstrap"],
                    bootstrap_iterations=options["bootstrap_n"],
                    include_complexity=options["include_complexity"],
                    include_advanced_metrics=options["include_advanced_metrics"],
                )

        # Store result
        st.session_state.analysis_results.append(result)
        st.success(f"✅ Анализ завершён! ({result.n_words} слов)")

        # Display results
        render_metrics(result)

        # Bootstrap info
        render_bootstrap_info(result)

        # Tabs for visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Распределение", "📈 По частоте", "📉 Закон Ципфа", "📋 Таблица", "📑 ВСЁ"]
        )

        with tab1:
            fig = create_letter_distribution_chart(result, sort_by="alphabet", show_cumulative=True)
            st.plotly_chart(fig, use_container_width=True, key=f"single_tab1_dist_{id(result)}")

        with tab2:
            fig = create_letter_distribution_chart(result, sort_by="frequency")
            st.plotly_chart(fig, use_container_width=True, key=f"single_tab2_dist_{id(result)}")

        with tab3:
            fig = create_zipf_plot(result)
            st.plotly_chart(fig, use_container_width=True, key=f"single_tab3_zipf_{id(result)}")

        with tab4:
            df = st.session_state.analyzer.to_dataframe(result)
            st.dataframe(df, use_container_width=True)
            
        with tab5:
            render_full_report(result)

        # Advanced metrics in expander
        with st.expander("🔬 Расширенные метрики", expanded=True):
            render_advanced_metrics(result, key_suffix="main_tab")

