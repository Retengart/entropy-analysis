"""
Batch analysis tab.
"""

import streamlit as st

from entropy_analysis.dashboard.components.metrics import render_full_report
from entropy_analysis.visualization.charts import (
    create_correlation_scatter,
    create_entropy_histogram,
)


def batch_analysis(options):
    """Batch analysis tab."""
    st.header("📁 Пакетный анализ")

    uploaded_files = st.file_uploader(
        "Загрузите файлы", type=["txt", "md"], accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Загружено файлов: {len(uploaded_files)}")

        if st.button("🔍 Анализировать все", type="primary"):
            texts = []

            progress = st.progress(0)
            status_container = st.empty()

            # Функция для логирования
            def log_callback(msg: str):
                if options.get("show_verbose_logs"):
                    with status_container.container():
                        st.text(f"⏳ {msg}")

            for i, file in enumerate(uploaded_files):
                file.seek(0)  # Reset file position
                raw_content = file.read()
                try:
                    content = raw_content.decode("utf-8")
                except UnicodeDecodeError:
                    content = raw_content.decode("cp1251")
                texts.append((file.name, content))
                progress.progress((i + 1) / len(uploaded_files))

            # Используем st.status для логов если включено
            if options.get("show_verbose_logs"):
                with st.status("Пакетный анализ...", expanded=True) as status:
                    def status_logger(msg: str):
                        status.write(msg)
                    
                    batch_result = st.session_state.analyzer.analyze_batch(
                        texts,
                        include_bootstrap=options["include_bootstrap"],
                        include_advanced_metrics=options["include_advanced_metrics"],
                        log_callback=status_logger,
                    )
                    status.update(label="Анализ завершен!", state="complete", expanded=False)
            else:
                with st.spinner("Анализируем тексты..."):
                    batch_result = st.session_state.analyzer.analyze_batch(
                        texts,
                        include_bootstrap=options["include_bootstrap"],
                        include_advanced_metrics=options["include_advanced_metrics"],
                    )

            st.success(f"✅ Проанализировано {len(batch_result.results)} текстов!")

            # Summary metrics
            if batch_result.extended_stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Среднее H", f"{batch_result.extended_stats.mean:.4f}")
                with col2:
                    st.metric("Медиана H", f"{batch_result.extended_stats.median:.4f}")
                with col3:
                    st.metric("Стд. откл.", f"{batch_result.extended_stats.std_dev:.4f}")
                with col4:
                    if batch_result.correlation is not None:
                        st.metric("Корреляция r", f"{batch_result.correlation:.4f}")

            # Visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Гистограмма", "📈 Корреляция", "📋 Таблица", "📑 ВСЁ"])

            with tab1:
                fig = create_entropy_histogram(batch_result)
                st.plotly_chart(fig, use_container_width=True, key=f"batch_tab1_hist_{id(batch_result)}")

            with tab2:
                fig = create_correlation_scatter(batch_result, label_top_n=5)
                st.plotly_chart(fig, use_container_width=True, key=f"batch_tab2_scatter_{id(batch_result)}")

            with tab3:
                df = st.session_state.analyzer.batch_to_dataframe(batch_result)
                st.dataframe(df, use_container_width=True)

                # Download button
                csv = df.write_csv()
                st.download_button("📥 Скачать CSV", csv, "entropy_analysis.csv", "text/csv")
                
            with tab4:
                st.header("Сводный отчет по пакетному анализу")
                
                st.markdown("### Общая статистика")
                if batch_result.extended_stats:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Среднее H", f"{batch_result.extended_stats.mean:.4f}")
                    col2.metric("Медиана H", f"{batch_result.extended_stats.median:.4f}")
                    col3.metric("Стд. откл.", f"{batch_result.extended_stats.std_dev:.4f}")
                
                st.markdown("### Детальные отчеты по текстам")
                for name, res in batch_result.results:
                    with st.expander(f"📄 {name}"):
                        render_full_report(res)

