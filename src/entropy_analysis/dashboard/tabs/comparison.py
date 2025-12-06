"""
Comparison tabs (two texts and two authors).
"""

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from entropy_analysis.core.stats import compare_groups_statistically
from entropy_analysis.dashboard.components.metrics import render_advanced_metrics, render_full_report, render_metrics
from entropy_analysis.core.metrics import compare_ngram_distributions
from entropy_analysis.visualization.charts import (
    create_correlation_scatter,
    create_dual_author_comparison,
    create_entropy_histogram,
    create_letter_distribution_chart,
    create_ngram_comparison_chart,
    create_radar_chart,
)


def text_comparison(options):
    """Text comparison tab."""
    st.header("🔄 Сравнение текстов")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Текст 1")
        text1 = st.text_area("Текст 1", height=150, key="text1")
        name1 = st.text_input("Название 1", "Текст 1")

    with col2:
        st.subheader("Текст 2")
        text2 = st.text_area("Текст 2", height=150, key="text2")
        name2 = st.text_input("Название 2", "Текст 2")

    if st.button("🔍 Сравнить", type="primary"):
        if not text1.strip() or not text2.strip():
            st.warning("Введите оба текста для сравнения")
            return

        status_container = st.empty()
        
        # Функция для логирования
        def log_callback(msg: str):
            if options.get("show_verbose_logs"):
                with status_container.container():
                    st.text(f"⏳ {msg}")

        # Используем st.status для логов если включено
        if options.get("show_verbose_logs"):
            with st.status("Сравнение текстов...", expanded=True) as status:
                def status_logger(msg: str):
                    status.write(msg)
                
                status_logger("Расчет расхождений (Kullback-Leibler, Jensen-Shannon, Delta)...")
                comparison = st.session_state.analyzer.compare(
                    text1, 
                    text2, 
                    name1, 
                    name2,
                    include_delta=options["include_advanced_metrics"],
                )

                status_logger(f"Анализ текста 1: {name1}...")
                result1 = st.session_state.analyzer.analyze(
                    text1, 
                    name1, 
                    include_advanced_metrics=options["include_advanced_metrics"],
                    log_callback=status_logger,
                )
                
                status_logger(f"Анализ текста 2: {name2}...")
                result2 = st.session_state.analyzer.analyze(
                    text2, 
                    name2, 
                    include_advanced_metrics=options["include_advanced_metrics"],
                    log_callback=status_logger,
                )
                status.update(label="Сравнение завершено!", state="complete", expanded=False)
        else:
            with st.spinner("Сравниваем тексты..."):
                comparison = st.session_state.analyzer.compare(
                    text1, 
                    text2, 
                    name1, 
                    name2,
                    include_delta=options["include_advanced_metrics"],
                )

                result1 = st.session_state.analyzer.analyze(
                    text1, name1, include_advanced_metrics=options["include_advanced_metrics"]
                )
                result2 = st.session_state.analyzer.analyze(
                    text2, name2, include_advanced_metrics=options["include_advanced_metrics"]
                )

        st.success("✅ Сравнение завершено!")

        # Comparison metrics - Divergences
        st.subheader("📊 Метрики расхождения")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "KL(P||Q)", 
                f"{comparison.kl_divergence_p_q:.4f}",
                help="KL-дивергенция (несимметричная). Показывает информационные потери.",
            )
        with col2:
            st.metric(
                "JS Divergence", 
                f"{comparison.js_divergence:.4f}",
                help="Jensen-Shannon дивергенция (симметричная, 0-1). Стандартная метрика.",
            )
        with col3:
            st.metric(
                "Cosine Similarity",
                f"{comparison.cosine_similarity:.4f}" if comparison.cosine_similarity else "—",
                help="Косинусное сходство распределений букв (0-1). Выше = похожее.",
            )
        
        # Information distances
        st.subheader("📏 Информационные расстояния")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Wasserstein",
                f"{comparison.wasserstein_distance:.4f}" if comparison.wasserstein_distance else "—",
                help="Расстояние Вассерштейна (Earth Mover's). Учитывает 'близость' категорий.",
            )
        with col2:
            st.metric(
                "Hellinger",
                f"{comparison.hellinger_distance:.4f}" if comparison.hellinger_distance else "—",
                help="Расстояние Хеллингера (0-1). Симметричное, устойчивое к нулям.",
            )
        with col3:
            st.metric(
                "Total Variation",
                f"{comparison.total_variation_distance:.4f}" if comparison.total_variation_distance else "—",
                help="Total Variation Distance (0-1). Максимальная разница вероятностей.",
            )

        # Delta Metrics (если включены)
        if comparison.burrows_delta is not None:
            st.subheader("🔍 Атрибуция (Delta)")
            st.info(f"Рассчитано на {comparison.mfw_used} самых частых словах (MFW)")
            st.warning("""
            ⚠️ **Ограничение**: Delta-метрики разработаны для сравнения текста с **корпусом** 
            (множеством текстов известных авторов). При сравнении только 2 текстов 
            Z-scores вычисляются упрощённо и результаты следует интерпретировать осторожно.
            Для надёжной атрибуции используйте вкладку "Сравнение авторов" с множеством текстов.
            """)
            
            d_col1, d_col2, d_col3 = st.columns(3)
            
            with d_col1:
                st.metric(
                    "Burrows' Delta",
                    f"{comparison.burrows_delta:.4f}",
                    help="Классическая Delta (Manhattan distance z-scores). Меньше = ближе стиль.",
                )
            with d_col2:
                st.metric(
                    "Cosine Delta",
                    f"{comparison.cosine_delta:.4f}",
                    help="Würzburg Delta (Cosine distance z-scores). State-of-the-art. Меньше = ближе стиль (0..2).",
                )
            with d_col3:
                st.metric(
                    "Eder's Delta",
                    f"{comparison.eders_delta:.4f}",
                    help="Взвешенная Delta. Дает больше веса частым словам.",
                )

        # Side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 {name1}")
            render_metrics(result1)
            fig = create_letter_distribution_chart(result1)
            st.plotly_chart(fig, use_container_width=True, key=f"compare_text1_dist_{id(result1)}")

        with col2:
            st.subheader(f"📊 {name2}")
            render_metrics(result2)
            fig = create_letter_distribution_chart(result2)
            st.plotly_chart(fig, use_container_width=True, key=f"compare_text2_dist_{id(result2)}")

        # Radar comparison
        st.subheader("🎯 Радарное сравнение")
        fig = create_radar_chart([result1, result2], [name1, name2])
        st.plotly_chart(fig, use_container_width=True, key=f"compare_radar_{id(result1)}_{id(result2)}")

        # Advanced metrics for each text
        st.subheader("🔬 Расширенные метрики")
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.markdown(f"**{name1}**")
            with st.expander("Показать расширенные метрики", expanded=False):
                render_advanced_metrics(result1, key_suffix=f"compare_adv1_{id(result1)}")
        
        with adv_col2:
            st.markdown(f"**{name2}**")
            with st.expander("Показать расширенные метрики", expanded=False):
                render_advanced_metrics(result2, key_suffix=f"compare_adv2_{id(result2)}")


def author_comparison(options):
    """Compare two authors using split analysis."""
    st.header("👥 Сравнение двух авторов")

    st.markdown("""
    Этот режим позволяет сравнить двух авторов, загрузив сборники их произведений.
    Каждое произведение должно быть отделено разделителем (например, `***`).

    **Рекомендуемые метрики для сравнения:**
    - 📈 Корреляция H vs N (паттерн автора)
    - 🔢 Perplexity (эффективное разнообразие)
    - 🎯 Evenness (равномерность стиля)
    - 📊 Средние метрики по всем текстам
    """)

    # Input for two authors
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Автор 1")
        author1_name = st.text_input("Имя автора 1", "Автор 1", key="author1_name")

        input_method1 = st.radio(
            "Способ ввода", ["Загрузить файл", "Вставить текст"], key="author1_method"
        )

        text1 = ""
        if input_method1 == "Загрузить файл":
            file1 = st.file_uploader(f"Файл для {author1_name}", type=["txt"], key="author1_file")
            if file1:
                file1.seek(0)
                raw = file1.read()
                try:
                    text1 = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text1 = raw.decode("cp1251")
                st.success(f"✓ Загружено {len(text1)} символов")
        else:
            text1 = st.text_area(
                "Текст с разделителями",
                height=200,
                key="author1_text",
                placeholder="Стих 1\n***\nСтих 2\n***\n...",
            )

    with col2:
        st.subheader("👤 Автор 2")
        author2_name = st.text_input("Имя автора 2", "Автор 2", key="author2_name")

        input_method2 = st.radio(
            "Способ ввода", ["Загрузить файл", "Вставить текст"], key="author2_method"
        )

        text2 = ""
        if input_method2 == "Загрузить файл":
            file2 = st.file_uploader(f"Файл для {author2_name}", type=["txt"], key="author2_file")
            if file2:
                file2.seek(0)
                raw = file2.read()
                try:
                    text2 = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text2 = raw.decode("cp1251")
                st.success(f"✓ Загружено {len(text2)} символов")
        else:
            text2 = st.text_area(
                "Текст с разделителями",
                height=200,
                key="author2_text",
                placeholder="Стих 1\n***\nСтих 2\n***\n...",
            )

    # Delimiter
    delimiter = st.text_input("Разделитель произведений", "***", key="comparison_delimiter")
    
    # Filter options
    with st.expander("⚙️ Настройки фильтрации", expanded=False):
        min_words = st.number_input(
            "Минимальное количество слов в сегменте",
            min_value=1,
            max_value=1000,
            value=10,
            step=1,
            help="Сегменты с меньшим количеством слов будут пропущены (помогает исключить аннотационные строки и заголовки)",
            key="comparison_min_words"
        )
    
    # Outlier detection options
    with st.expander("🔍 Настройки определения выбросов", expanded=False):
        st.markdown("""
        **Методы определения выбросов:**
        - **IQR** (по умолчанию): Использует межквартильный размах. Хорош для нормальных распределений.
        - **Z-score**: Использует стандартное отклонение. Эффективен для больших выборок.
        - **Modified Z-score**: Использует медианное абсолютное отклонение (MAD). Устойчив к выбросам.
        """)
        
        outlier_method = st.selectbox(
            "Метод определения выбросов",
            options=["iqr", "zscore", "modified_zscore"],
            index=0,
            help="Выберите метод для определения выбросов на графиках",
            key="comparison_outlier_method"
        )
        
        # Default thresholds for each method
        default_thresholds = {
            "iqr": 1.5,
            "zscore": 3.0,
            "modified_zscore": 3.5,
        }
        
        threshold_help = {
            "iqr": "Множитель IQR (по умолчанию 1.5). Больше значение = меньше выбросов",
            "zscore": "Порог Z-score (по умолчанию 3.0). Больше значение = меньше выбросов",
            "modified_zscore": "Порог Modified Z-score (по умолчанию 3.5). Больше значение = меньше выбросов",
        }
        
        outlier_threshold = st.number_input(
            f"Порог для {outlier_method.upper()}",
            min_value=0.1,
            max_value=10.0,
            value=default_thresholds[outlier_method],
            step=0.1,
            help=threshold_help[outlier_method],
            key="comparison_outlier_threshold"
        )
        
        highlight_outliers = st.checkbox(
            "Выделять выбросы на графиках",
            value=True,
            help="Если отключено, выбросы не будут выделяться отдельным цветом",
            key="comparison_highlight_outliers"
        )

    if st.button("🔍 Сравнить авторов", type="primary", key="compare_authors_btn"):
        if not text1.strip() or not text2.strip():
            st.error("❌ Загрузите тексты обоих авторов")
            return

        status_container = st.empty()
        
        # Функция для логирования
        def log_callback(msg: str):
            if options.get("show_verbose_logs"):
                with status_container.container():
                    st.text(f"⏳ {msg}")

        # Используем st.status для логов если включено
        if options.get("show_verbose_logs"):
            with st.status(f"Анализ авторов...", expanded=True) as status:
                def status_logger(msg: str):
                    status.write(msg)
                
                status_logger(f"Анализ автора 1: {author1_name}...")
                batch1 = st.session_state.analyzer.split_and_analyze(
                    text1,
                    delimiter=delimiter,
                    auto_name=True,
                    include_advanced_metrics=options["include_advanced_metrics"],
                    min_segment_words=min_words,
                    log_callback=status_logger,
                )
                
                status_logger(f"Анализ автора 2: {author2_name}...")
                batch2 = st.session_state.analyzer.split_and_analyze(
                    text2,
                    delimiter=delimiter,
                    auto_name=True,
                    include_advanced_metrics=options["include_advanced_metrics"],
                    min_segment_words=min_words,
                    log_callback=status_logger,
                )
                status.update(label="Анализ авторов завершен!", state="complete", expanded=False)
        else:
            with st.spinner(f"Анализируем {author1_name} и {author2_name}..."):
                batch1 = st.session_state.analyzer.split_and_analyze(
                    text1,
                    delimiter=delimiter,
                    auto_name=True,
                    include_advanced_metrics=options["include_advanced_metrics"],
                    min_segment_words=min_words,
                )
                batch2 = st.session_state.analyzer.split_and_analyze(
                    text2,
                    delimiter=delimiter,
                    auto_name=True,
                    include_advanced_metrics=options["include_advanced_metrics"],
                    min_segment_words=min_words,
                )

        n1 = len(batch1.results)
        n2 = len(batch2.results)

        if n1 == 0 or n2 == 0:
            st.error(f"❌ Не найдено произведений. Проверьте разделитель '{delimiter}'")
            return

        st.success(
            f"✅ Проанализировано: {author1_name} - {n1} произв., {author2_name} - {n2} произв."
        )

        # Предупреждение о минимальном размере выборки
        min_recommended = 20
        if n1 < min_recommended or n2 < min_recommended:
            st.warning(f"""
            ⚠️ **Малый размер выборки**: Для надёжных статистических выводов рекомендуется 
            минимум **{min_recommended} текстов** на автора. 
            Текущее: {author1_name} = {n1}, {author2_name} = {n2}.
            
            С малыми выборками:
            - P-values могут быть ненадёжными
            - Доверительные интервалы будут широкими
            - Effect size (Cohen's d) наиболее информативен
            """)

        # ========================================================================
        # MAIN COMPARISON SECTION
        # ========================================================================

        st.header("📊 Результаты сравнения")

        # Key metrics comparison
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            [
                "🎯 Ключевые различия",
                "📊 Статистические тесты",
                "📈 Корреляция H vs N",
                "🔤 Сравнение N-грамм",
                "🔬 Детальные метрики",
                "📊 Распределения",
                "📋 Таблицы",
                "📑 ВСЁ",
            ]
        )

        with tab1:
            st.subheader("🎯 Ключевые различия между авторами")

            # Calculate average metrics
            def calc_avg_metrics(batch):
                results = batch.results
                avg = {
                    "h": np.mean([r.shannon_entropy for _, r in results if r.shannon_entropy]),
                    "n": np.mean([r.n_words for _, r in results]),
                    "perp": np.mean([r.enhanced.perplexity for _, r in results if r.enhanced]),
                    "even": np.mean([r.enhanced.evenness for _, r in results if r.enhanced]),
                    "util": np.mean(
                        [r.enhanced.alphabet_utilization for _, r in results if r.enhanced]
                    ),
                    "hhi": np.mean([r.enhanced.herfindahl_index for _, r in results if r.enhanced]),
                    "redun": np.mean([r.enhanced.redundancy for _, r in results if r.enhanced]),
                }
                return avg

            avg1 = calc_avg_metrics(batch1)
            avg2 = calc_avg_metrics(batch2)

            # Comparison cards
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("📈 **Средняя энтропия**")
                sub1, sub2 = st.columns(2)
                sub1.metric(author1_name, f"{avg1['h']:.3f}")
                sub2.metric(
                    author2_name,
                    f"{avg2['h']:.3f}",
                    delta=f"{avg2['h'] - avg1['h']:.3f}",
                    delta_color="normal",
                )

            with col2:
                st.markdown("🔢 **Средняя Perplexity**")
                sub1, sub2 = st.columns(2)
                sub1.metric(author1_name, f"{avg1['perp']:.2f}")
                sub2.metric(
                    author2_name,
                    f"{avg2['perp']:.2f}",
                    delta=f"{avg2['perp'] - avg1['perp']:.2f}",
                    delta_color="normal",
                )

            with col3:
                st.markdown("🎯 **Средняя Evenness**")
                sub1, sub2 = st.columns(2)
                sub1.metric(author1_name, f"{avg1['even']:.3f}")
                sub2.metric(
                    author2_name,
                    f"{avg2['even']:.3f}",
                    delta=f"{avg2['even'] - avg1['even']:.3f}",
                    delta_color="normal",
                )

            # Correlation comparison
            st.subheader("📉 Сравнение корреляций H vs N")
            st.markdown("**Это самая важная метрика для различия авторских стилей!**")

            col1, col2, col3 = st.columns(3)

            with col1:
                if batch1.correlation is not None:
                    st.metric(
                        f"r ({author1_name})",
                        f"{batch1.correlation:.4f}",
                        help="Коэффициент корреляции",
                    )
                if batch2.correlation is not None:
                    st.metric(f"r ({author2_name})", f"{batch2.correlation:.4f}")

            with col2:
                if batch1.correlation_slope is not None:
                    st.metric(
                        f"Slope ({author1_name})",
                        f"{batch1.correlation_slope:.6f}",
                        help="Наклон регрессии",
                    )
                if batch2.correlation_slope is not None:
                    st.metric(f"Slope ({author2_name})", f"{batch2.correlation_slope:.6f}")

            with col3:
                if batch1.correlation_r_squared is not None:
                    st.metric(
                        f"R² ({author1_name})",
                        f"{batch1.correlation_r_squared:.4f}",
                        help="Коэффициент детерминации",
                    )
                if batch2.correlation_r_squared is not None:
                    st.metric(f"R² ({author2_name})", f"{batch2.correlation_r_squared:.4f}")

            # Interpretation
            st.subheader("💡 Интерпретация")

            if batch1.correlation and batch2.correlation:
                diff_r = abs(batch1.correlation - batch2.correlation)
                diff_slope = abs(batch1.correlation_slope - batch2.correlation_slope)

                if diff_r > 0.2:
                    st.info(f"🔍 **Сильное различие в корреляции** (Δr = {diff_r:.3f})")
                elif diff_r > 0.1:
                    st.info(f"🔍 **Умеренное различие в корреляции** (Δr = {diff_r:.3f})")
                else:
                    st.info(f"🔍 **Слабое различие в корреляции** (Δr = {diff_r:.3f})")

                if diff_slope > 0.001:
                    st.info(f"📐 **Различие в наклоне регрессии** (Δslope = {diff_slope:.6f})")
                    st.caption(
                        "Разный наклон означает разную скорость роста энтропии с длиной текста"
                    )

        with tab2:
            st.subheader("📊 Статистические тесты значимости различий")

            if n1 < 5 or n2 < 5:
                st.warning(
                    f"⚠️ **Малый размер выборки** ({n1} и {n2} произведений). "
                    "Результаты статистических тестов могут быть недостоверны. "
                    "Для надежных выводов рекомендуется анализировать не менее 5-10 текстов каждого автора."
                )

            st.markdown("""
            **Самые строгие статистические тесты для максимальной объективности!**

            Проверяем гипотезу: *Действительно ли авторы различаются, или это случайность?*
            """)

            # Import statistical functions
            from entropy_analysis.core.stats import (
                compare_groups_statistically,
                correct_multiple_comparisons,
            )

            # Extract metric arrays
            h1 = np.array([r.shannon_entropy for _, r in batch1.results if r.shannon_entropy])
            h2 = np.array([r.shannon_entropy for _, r in batch2.results if r.shannon_entropy])

            perp1 = np.array([r.enhanced.perplexity for _, r in batch1.results if r.enhanced])
            perp2 = np.array([r.enhanced.perplexity for _, r in batch2.results if r.enhanced])

            even1 = np.array([r.enhanced.evenness for _, r in batch1.results if r.enhanced])
            even2 = np.array([r.enhanced.evenness for _, r in batch2.results if r.enhanced])

            # Letter-level n-gram entropies
            letter_bigram1 = np.array([r.letter_bigram_entropy for _, r in batch1.results if r.letter_bigram_entropy is not None])
            letter_bigram2 = np.array([r.letter_bigram_entropy for _, r in batch2.results if r.letter_bigram_entropy is not None])
            
            letter_trigram1 = np.array([r.letter_trigram_entropy for _, r in batch1.results if r.letter_trigram_entropy is not None])
            letter_trigram2 = np.array([r.letter_trigram_entropy for _, r in batch2.results if r.letter_trigram_entropy is not None])

            # === ENTROPY COMPARISON ===
            st.subheader("1️⃣ Энтропия (H)")

            comp_h = compare_groups_statistically(h1, h2, n_permutations=5000, n_bootstrap=3000)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Различие",
                    f"{comp_h.permutation_test.observed_difference:.4f}",
                    help="Разница средних значений",
                )

                # Significance indicator
                if comp_h.permutation_test.is_significant:
                    st.success("✅ **Статистически значимо!**")
                else:
                    st.info("⚪ Незначимо (p ≥ 0.05)")

            with col2:
                st.metric(
                    "p-value (Permutation Test)",
                    f"{comp_h.permutation_test.p_value:.4f}",
                    help="Вероятность случайного различия",
                )

                if comp_h.permutation_test.p_value < 0.001:
                    st.caption("⭐⭐⭐ p < 0.001 (очень значимо)")
                elif comp_h.permutation_test.p_value < 0.01:
                    st.caption("⭐⭐ p < 0.01 (значимо)")
                elif comp_h.permutation_test.p_value < 0.05:
                    st.caption("⭐ p < 0.05 (значимо)")
                else:
                    st.caption("p ≥ 0.05 (незначимо)")

            with col3:
                st.metric(
                    "Effect Size (Cohen's d)",
                    f"{comp_h.effect_size.cohens_d:.3f}",
                    help="Размер эффекта",
                )

                magnitude = comp_h.effect_size.effect_magnitude
                if magnitude == "large":
                    st.caption("🔵 Большой эффект (d > 0.8)")
                elif magnitude == "medium":
                    st.caption("🟢 Средний эффект (d > 0.5)")
                elif magnitude == "small":
                    st.caption("🟡 Малый эффект (d > 0.2)")
                else:
                    st.caption("⚪ Незначительный (d < 0.2)")

            # Bootstrap CI
            st.markdown(
                f"**95% доверительный интервал для различия:** "
                f"[{comp_h.bootstrap_diff.ci_lower:.4f}, {comp_h.bootstrap_diff.ci_upper:.4f}]"
            )

            if comp_h.bootstrap_diff.is_significant:
                st.success("✅ 0 не входит в доверительный интервал → различие значимо!")
            else:
                st.info("⚪ 0 входит в доверительный интервал → различие незначимо")

            st.divider()

            # === PERPLEXITY COMPARISON ===
            st.subheader("2️⃣ Perplexity (Разнообразие)")

            comp_perp = compare_groups_statistically(
                perp1, perp2, n_permutations=5000, n_bootstrap=3000
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Различие",
                    f"{comp_perp.permutation_test.observed_difference:.3f}",
                )

                if comp_perp.permutation_test.is_significant:
                    st.success("✅ **Статистически значимо!**")
                else:
                    st.info("⚪ Незначимо")

            with col2:
                st.metric(
                    "p-value",
                    f"{comp_perp.permutation_test.p_value:.4f}",
                )

            with col3:
                st.metric(
                    "Cohen's d",
                    f"{comp_perp.effect_size.cohens_d:.3f}",
                )
                st.caption(f"{comp_perp.effect_size.effect_magnitude.title()} эффект")

            st.markdown(
                f"**95% CI:** [{comp_perp.bootstrap_diff.ci_lower:.3f}, {comp_perp.bootstrap_diff.ci_upper:.3f}]"
            )

            st.divider()

            # === EVENNESS COMPARISON ===
            st.subheader("3️⃣ Evenness (Равномерность)")

            comp_even = compare_groups_statistically(
                even1, even2, n_permutations=5000, n_bootstrap=3000
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Различие",
                    f"{comp_even.permutation_test.observed_difference:.4f}",
                )

                if comp_even.permutation_test.is_significant:
                    st.success("✅ **Статистически значимо!**")
                else:
                    st.info("⚪ Незначимо")

            with col2:
                st.metric(
                    "p-value",
                    f"{comp_even.permutation_test.p_value:.4f}",
                )

            with col3:
                st.metric(
                    "Cohen's d",
                    f"{comp_even.effect_size.cohens_d:.3f}",
                )
                st.caption(f"{comp_even.effect_size.effect_magnitude.title()} эффект")

            st.markdown(
                f"**95% CI:** [{comp_even.bootstrap_diff.ci_lower:.4f}, {comp_even.bootstrap_diff.ci_upper:.4f}]"
            )

            st.divider()

            # === LETTER BIGRAM ENTROPY COMPARISON ===
            if len(letter_bigram1) > 0 and len(letter_bigram2) > 0:
                st.subheader("4️⃣ Энтропия биграмм букв (H₂)")

                st.markdown("""
                **Глубокая метрика**: Анализирует последовательности из двух букв подряд.
                Показывает, насколько предсказуемы сочетания букв в тексте автора.
                """)

                comp_letter_bigram = compare_groups_statistically(
                    letter_bigram1, letter_bigram2, n_permutations=5000, n_bootstrap=3000
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Различие",
                        f"{comp_letter_bigram.permutation_test.observed_difference:.4f}",
                    )

                    if comp_letter_bigram.permutation_test.is_significant:
                        st.success("✅ **Статистически значимо!**")
                    else:
                        st.info("⚪ Незначимо")

                with col2:
                    st.metric(
                        "p-value",
                        f"{comp_letter_bigram.permutation_test.p_value:.4f}",
                    )

                with col3:
                    st.metric(
                        "Cohen's d",
                        f"{comp_letter_bigram.effect_size.cohens_d:.3f}",
                    )
                    st.caption(f"{comp_letter_bigram.effect_size.effect_magnitude.title()} эффект")

                st.markdown(
                    f"**95% CI:** [{comp_letter_bigram.bootstrap_diff.ci_lower:.4f}, {comp_letter_bigram.bootstrap_diff.ci_upper:.4f}]"
                )

                st.divider()

            # === LETTER TRIGRAM ENTROPY COMPARISON ===
            if len(letter_trigram1) > 0 and len(letter_trigram2) > 0:
                st.subheader("5️⃣ Энтропия триграмм букв (H₃)")

                st.markdown("""
                **Глубокая метрика**: Анализирует последовательности из трёх букв подряд.
                Показывает контекстные зависимости и стилистические паттерны автора.
                """)

                comp_letter_trigram = compare_groups_statistically(
                    letter_trigram1, letter_trigram2, n_permutations=5000, n_bootstrap=3000
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Различие",
                        f"{comp_letter_trigram.permutation_test.observed_difference:.4f}",
                    )

                    if comp_letter_trigram.permutation_test.is_significant:
                        st.success("✅ **Статистически значимо!**")
                    else:
                        st.info("⚪ Незначимо")

                with col2:
                    st.metric(
                        "p-value",
                        f"{comp_letter_trigram.permutation_test.p_value:.4f}",
                    )

                with col3:
                    st.metric(
                        "Cohen's d",
                        f"{comp_letter_trigram.effect_size.cohens_d:.3f}",
                    )
                    st.caption(f"{comp_letter_trigram.effect_size.effect_magnitude.title()} эффект")

                st.markdown(
                    f"**95% CI:** [{comp_letter_trigram.bootstrap_diff.ci_lower:.4f}, {comp_letter_trigram.bootstrap_diff.ci_upper:.4f}]"
                )

                st.divider()

            # === SUMMARY TABLE ===
            st.subheader("📋 Сводная таблица статистических тестов")

            # Build summary table dynamically
            metrics_list = ["Энтропия H", "Perplexity", "Evenness"]
            differences_list = [
                f"{comp_h.permutation_test.observed_difference:.4f}",
                f"{comp_perp.permutation_test.observed_difference:.3f}",
                f"{comp_even.permutation_test.observed_difference:.4f}",
            ]
            pvalues_list = [
                f"{comp_h.permutation_test.p_value:.4f}",
                f"{comp_perp.permutation_test.p_value:.4f}",
                f"{comp_even.permutation_test.p_value:.4f}",
            ]
            cohens_d_list = [
                f"{comp_h.effect_size.cohens_d:.3f}",
                f"{comp_perp.effect_size.cohens_d:.3f}",
                f"{comp_even.effect_size.cohens_d:.3f}",
            ]
            effects_list = [
                comp_h.effect_size.effect_magnitude.title(),
                comp_perp.effect_size.effect_magnitude.title(),
                comp_even.effect_size.effect_magnitude.title(),
            ]
            significant_list = [
                "✅ Да" if comp_h.permutation_test.is_significant else "⚪ Нет",
                "✅ Да" if comp_perp.permutation_test.is_significant else "⚪ Нет",
                "✅ Да" if comp_even.permutation_test.is_significant else "⚪ Нет",
            ]

            # Add letter-level metrics if available
            if len(letter_bigram1) > 0 and len(letter_bigram2) > 0:
                metrics_list.append("Энтропия биграмм букв (H₂)")
                differences_list.append(f"{comp_letter_bigram.permutation_test.observed_difference:.4f}")
                pvalues_list.append(f"{comp_letter_bigram.permutation_test.p_value:.4f}")
                cohens_d_list.append(f"{comp_letter_bigram.effect_size.cohens_d:.3f}")
                effects_list.append(comp_letter_bigram.effect_size.effect_magnitude.title())
                significant_list.append("✅ Да" if comp_letter_bigram.permutation_test.is_significant else "⚪ Нет")

            if len(letter_trigram1) > 0 and len(letter_trigram2) > 0:
                metrics_list.append("Энтропия триграмм букв (H₃)")
                differences_list.append(f"{comp_letter_trigram.permutation_test.observed_difference:.4f}")
                pvalues_list.append(f"{comp_letter_trigram.permutation_test.p_value:.4f}")
                cohens_d_list.append(f"{comp_letter_trigram.effect_size.cohens_d:.3f}")
                effects_list.append(comp_letter_trigram.effect_size.effect_magnitude.title())
                significant_list.append("✅ Да" if comp_letter_trigram.permutation_test.is_significant else "⚪ Нет")

            # Apply multiple comparison correction
            p_values_raw = [
                comp_h.permutation_test.p_value,
                comp_perp.permutation_test.p_value,
                comp_even.permutation_test.p_value,
            ]
            if len(letter_bigram1) > 0 and len(letter_bigram2) > 0:
                p_values_raw.append(comp_letter_bigram.permutation_test.p_value)
            if len(letter_trigram1) > 0 and len(letter_trigram2) > 0:
                p_values_raw.append(comp_letter_trigram.permutation_test.p_value)
            
            # Apply FDR correction (Benjamini-Hochberg)
            correction = correct_multiple_comparisons(p_values_raw, method="fdr_bh", alpha=0.05)
            
            # Update p-values list with corrected values
            pvalues_corrected = correction.corrected_p_values[:len(pvalues_list)]
            pvalues_corrected_str = [f"{p:.4f}" for p in pvalues_corrected]
            
            # Add corrected significance column
            significant_corrected_list = [
                "✅ Да" if p < 0.05 else "⚪ Нет" for p in pvalues_corrected
            ]
            
            summary_data = {
                "Метрика": metrics_list,
                "Различие": differences_list,
                "p-value (raw)": pvalues_list,
                "p-value (FDR corrected)": pvalues_corrected_str,
                "Cohen's d": cohens_d_list,
                "Эффект": effects_list,
                "Значимо? (raw)": significant_list,
                "Значимо? (corrected)": significant_corrected_list,
            }

            df_summary = pl.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Show correction info
            st.info(f"""
            **Коррекция множественных сравнений (FDR-BH):**
            - Выполнено тестов: {correction.n_tests}
            - Значимых до коррекции: {correction.n_significant_original}
            - Значимых после коррекции: {correction.n_significant_corrected}
            
            ⚠️ **Важно:** При множественных сравнениях используйте скорректированные p-values для достоверных выводов.
            """)

            st.divider()

            # === INTERPRETATION ===
            st.subheader("💡 Интерпретация")

            # Count significant differences (including letter-level metrics)
            significant_metrics = [
                comp_h.permutation_test.is_significant,
                comp_perp.permutation_test.is_significant,
                comp_even.permutation_test.is_significant,
            ]
            
            if len(letter_bigram1) > 0 and len(letter_bigram2) > 0:
                significant_metrics.append(comp_letter_bigram.permutation_test.is_significant)
            if len(letter_trigram1) > 0 and len(letter_trigram2) > 0:
                significant_metrics.append(comp_letter_trigram.permutation_test.is_significant)
            
            n_significant = sum(significant_metrics)
            total_metrics = len(significant_metrics)

            if n_significant == total_metrics:
                st.success(f"""
                **🎯 Авторы значимо различаются по всем {total_metrics} метрикам!**

                Это означает, что {author1_name} и {author2_name} имеют **объективно разные** стили письма.
                Различия не являются случайными (p < 0.05 для всех метрик).
                
                {"✨ **Глубокий анализ подтверждает различия** на уровне буквенных последовательностей!" if total_metrics > 3 else ""}
                """)
            elif n_significant >= total_metrics * 0.6:  # 60% or more
                st.info(f"""
                **📊 Авторы значимо различаются по {n_significant} из {total_metrics} метрик.**

                Есть статистически подтверждённые различия в стилях.
                {"Включая различия на уровне буквенных последовательностей!" if any([
                    len(letter_bigram1) > 0 and len(letter_bigram2) > 0 and comp_letter_bigram.permutation_test.is_significant,
                    len(letter_trigram1) > 0 and len(letter_trigram2) > 0 and comp_letter_trigram.permutation_test.is_significant,
                ]) else ""}
                """)
            elif n_significant > 0:
                st.warning(f"""
                **⚠️ Авторы различаются только по {n_significant} из {total_metrics} метрик.**

                Различия частичные, авторы в целом похожи.
                {"Однако обнаружены различия на уровне буквенных последовательностей!" if any([
                    len(letter_bigram1) > 0 and len(letter_bigram2) > 0 and comp_letter_bigram.permutation_test.is_significant,
                    len(letter_trigram1) > 0 and len(letter_trigram2) > 0 and comp_letter_trigram.permutation_test.is_significant,
                ]) else ""}
                """)
            else:
                st.error("""
                **⚪ Нет статистически значимых различий.**

                По этим метрикам авторы неотличимы друг от друга (все p ≥ 0.05).
                """)

            # Effect sizes interpretation
            st.markdown("**Размеры эффектов (Cohen's d):**")

            large_effects = []
            if abs(comp_h.effect_size.cohens_d) > 0.8:
                large_effects.append("Энтропия")
            if abs(comp_perp.effect_size.cohens_d) > 0.8:
                large_effects.append("Perplexity")
            if abs(comp_even.effect_size.cohens_d) > 0.8:
                large_effects.append("Evenness")
            if len(letter_bigram1) > 0 and len(letter_bigram2) > 0 and abs(comp_letter_bigram.effect_size.cohens_d) > 0.8:
                large_effects.append("Энтропия биграмм букв")
            if len(letter_trigram1) > 0 and len(letter_trigram2) > 0 and abs(comp_letter_trigram.effect_size.cohens_d) > 0.8:
                large_effects.append("Энтропия триграмм букв")

            if large_effects:
                st.success(f"🔵 **Большие эффекты** (d > 0.8): {', '.join(large_effects)}")
            else:
                st.info("Все эффекты малые или средние")

        with tab3:
            st.subheader("📈 График корреляции H vs N")

            # Dual author comparison chart
            fig = create_dual_author_comparison(batch1, batch2, author1_name, author2_name)
            st.plotly_chart(fig, use_container_width=True, key=f"author_compare_dual_{id(batch1)}_{id(batch2)}")

            # Individual charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{author1_name}**")
                fig1 = create_correlation_scatter(
                    batch1, 
                    show_trendline=True,
                    highlight_outliers=highlight_outliers,
                    outlier_method=outlier_method,
                    outlier_threshold=outlier_threshold,
                )
                fig1.update_layout(height=400, title=f"{author1_name}: H vs N")
                st.plotly_chart(fig1, use_container_width=True, key=f"author_compare_scatter1_{id(batch1)}")

            with col2:
                st.markdown(f"**{author2_name}**")
                fig2 = create_correlation_scatter(
                    batch2, 
                    show_trendline=True,
                    highlight_outliers=highlight_outliers,
                    outlier_method=outlier_method,
                    outlier_threshold=outlier_threshold,
                )
                fig2.update_layout(height=400, title=f"{author2_name}: H vs N")
                st.plotly_chart(fig2, use_container_width=True, key=f"author_compare_scatter2_{id(batch2)}")

        with tab4:
            st.subheader("🔤 Сравнение распределений N-грамм букв")
            
            st.markdown("""
            **Глубокий анализ характерных последовательностей букв для каждого автора.**
            
            Этот анализ показывает:
            - Какие биграммы/триграммы наиболее характерны для каждого автора
            - Насколько различаются распределения n-грамм
            - Уникальные и общие n-граммы
            """)
            
            # Collect all letters from all texts for each author
            all_letters1: list[str] = []
            all_letters2: list[str] = []
            
            for _, result in batch1.results:
                if result.letter_bigram_distribution:
                    # We need to get the raw letters - let's use a workaround
                    # by collecting from segment names or re-extracting
                    pass
            
            # Use the normalizer to extract letters from original texts
            # Since we don't have the original texts here, we'll aggregate from results
            # Instead, let's compare average distributions
            
            # Compare bigram distributions if available
            st.markdown("### Биграммы букв")
            
            # Get average bigram stats from both authors
            bigram_stats1 = []
            bigram_stats2 = []
            
            for _, result in batch1.results:
                if result.letter_bigram_distribution:
                    bigram_stats1.append({
                        "entropy": result.letter_bigram_distribution.entropy,
                        "conditional": result.letter_bigram_distribution.conditional_entropy,
                        "unique": result.letter_bigram_distribution.unique_ngrams,
                        "hapax_ratio": result.letter_bigram_distribution.hapax_ratio,
                        "coverage_10": result.letter_bigram_distribution.coverage_top_10,
                        "zipf_alpha": result.letter_bigram_distribution.zipf_alpha,
                    })
            
            for _, result in batch2.results:
                if result.letter_bigram_distribution:
                    bigram_stats2.append({
                        "entropy": result.letter_bigram_distribution.entropy,
                        "conditional": result.letter_bigram_distribution.conditional_entropy,
                        "unique": result.letter_bigram_distribution.unique_ngrams,
                        "hapax_ratio": result.letter_bigram_distribution.hapax_ratio,
                        "coverage_10": result.letter_bigram_distribution.coverage_top_10,
                        "zipf_alpha": result.letter_bigram_distribution.zipf_alpha,
                    })
            
            if bigram_stats1 and bigram_stats2:
                # Calculate averages
                avg_bigram1 = {k: np.mean([s[k] for s in bigram_stats1]) for k in bigram_stats1[0].keys()}
                avg_bigram2 = {k: np.mean([s[k] for s in bigram_stats2]) for k in bigram_stats2[0].keys()}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{author1_name}**")
                    st.metric("Сред. энтропия биграмм", f"{avg_bigram1['entropy']:.3f} бит")
                    st.metric("Сред. условная H(X₂|X₁)", f"{avg_bigram1['conditional']:.3f} бит")
                    st.metric("Сред. Hapax ratio", f"{avg_bigram1['hapax_ratio'] * 100:.1f}%")
                    st.metric("Сред. покрытие топ-10", f"{avg_bigram1['coverage_10'] * 100:.1f}%")
                    st.metric("Сред. Zipf α", f"{avg_bigram1['zipf_alpha']:.2f}")
                
                with col2:
                    st.markdown(f"**{author2_name}**")
                    diff_h = avg_bigram2['entropy'] - avg_bigram1['entropy']
                    st.metric("Сред. энтропия биграмм", f"{avg_bigram2['entropy']:.3f} бит", delta=f"{diff_h:+.3f}")
                    diff_cond = avg_bigram2['conditional'] - avg_bigram1['conditional']
                    st.metric("Сред. условная H(X₂|X₁)", f"{avg_bigram2['conditional']:.3f} бит", delta=f"{diff_cond:+.3f}")
                    diff_hapax = (avg_bigram2['hapax_ratio'] - avg_bigram1['hapax_ratio']) * 100
                    st.metric("Сред. Hapax ratio", f"{avg_bigram2['hapax_ratio'] * 100:.1f}%", delta=f"{diff_hapax:+.1f}%")
                    diff_cov = (avg_bigram2['coverage_10'] - avg_bigram1['coverage_10']) * 100
                    st.metric("Сред. покрытие топ-10", f"{avg_bigram2['coverage_10'] * 100:.1f}%", delta=f"{diff_cov:+.1f}%")
                    diff_zipf = avg_bigram2['zipf_alpha'] - avg_bigram1['zipf_alpha']
                    st.metric("Сред. Zipf α", f"{avg_bigram2['zipf_alpha']:.2f}", delta=f"{diff_zipf:+.2f}")
                
                # Statistical comparison of bigram entropy
                bigram_h1 = np.array([s["entropy"] for s in bigram_stats1])
                bigram_h2 = np.array([s["entropy"] for s in bigram_stats2])
                
                if len(bigram_h1) >= 5 and len(bigram_h2) >= 5:
                    st.divider()
                    st.markdown("**Статистический тест различия энтропии биграмм:**")
                    
                    comp_bigram_h = compare_groups_statistically(bigram_h1, bigram_h2, n_permutations=3000, n_bootstrap=2000)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Различие", f"{comp_bigram_h.permutation_test.observed_difference:.4f}")
                    with col2:
                        st.metric("p-value", f"{comp_bigram_h.permutation_test.p_value:.4f}")
                    with col3:
                        st.metric("Cohen's d", f"{comp_bigram_h.effect_size.cohens_d:.3f}")
                    
                    if comp_bigram_h.permutation_test.is_significant:
                        st.success("✅ Различие в энтропии биграмм статистически значимо!")
                    else:
                        st.info("⚪ Различие в энтропии биграмм незначимо")
            else:
                st.warning("Недостаточно данных для сравнения биграмм")
            
            st.divider()
            
            # Compare trigram distributions
            st.markdown("### Триграммы букв")
            
            trigram_stats1 = []
            trigram_stats2 = []
            
            for _, result in batch1.results:
                if result.letter_trigram_distribution:
                    trigram_stats1.append({
                        "entropy": result.letter_trigram_distribution.entropy,
                        "conditional": result.letter_trigram_distribution.conditional_entropy,
                        "unique": result.letter_trigram_distribution.unique_ngrams,
                        "hapax_ratio": result.letter_trigram_distribution.hapax_ratio,
                        "coverage_10": result.letter_trigram_distribution.coverage_top_10,
                        "zipf_alpha": result.letter_trigram_distribution.zipf_alpha,
                    })
            
            for _, result in batch2.results:
                if result.letter_trigram_distribution:
                    trigram_stats2.append({
                        "entropy": result.letter_trigram_distribution.entropy,
                        "conditional": result.letter_trigram_distribution.conditional_entropy,
                        "unique": result.letter_trigram_distribution.unique_ngrams,
                        "hapax_ratio": result.letter_trigram_distribution.hapax_ratio,
                        "coverage_10": result.letter_trigram_distribution.coverage_top_10,
                        "zipf_alpha": result.letter_trigram_distribution.zipf_alpha,
                    })
            
            if trigram_stats1 and trigram_stats2:
                avg_trigram1 = {k: np.mean([s[k] for s in trigram_stats1]) for k in trigram_stats1[0].keys()}
                avg_trigram2 = {k: np.mean([s[k] for s in trigram_stats2]) for k in trigram_stats2[0].keys()}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{author1_name}**")
                    st.metric("Сред. энтропия триграмм", f"{avg_trigram1['entropy']:.3f} бит")
                    st.metric("Сред. условная H(X₃|X₁X₂)", f"{avg_trigram1['conditional']:.3f} бит")
                    st.metric("Сред. Hapax ratio", f"{avg_trigram1['hapax_ratio'] * 100:.1f}%")
                
                with col2:
                    st.markdown(f"**{author2_name}**")
                    diff_h = avg_trigram2['entropy'] - avg_trigram1['entropy']
                    st.metric("Сред. энтропия триграмм", f"{avg_trigram2['entropy']:.3f} бит", delta=f"{diff_h:+.3f}")
                    diff_cond = avg_trigram2['conditional'] - avg_trigram1['conditional']
                    st.metric("Сред. условная H(X₃|X₁X₂)", f"{avg_trigram2['conditional']:.3f} бит", delta=f"{diff_cond:+.3f}")
                    diff_hapax = (avg_trigram2['hapax_ratio'] - avg_trigram1['hapax_ratio']) * 100
                    st.metric("Сред. Hapax ratio", f"{avg_trigram2['hapax_ratio'] * 100:.1f}%", delta=f"{diff_hapax:+.1f}%")
                
                # Statistical comparison of trigram entropy
                trigram_h1 = np.array([s["entropy"] for s in trigram_stats1])
                trigram_h2 = np.array([s["entropy"] for s in trigram_stats2])
                
                if len(trigram_h1) >= 5 and len(trigram_h2) >= 5:
                    st.divider()
                    st.markdown("**Статистический тест различия энтропии триграмм:**")
                    
                    comp_trigram_h = compare_groups_statistically(trigram_h1, trigram_h2, n_permutations=3000, n_bootstrap=2000)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Различие", f"{comp_trigram_h.permutation_test.observed_difference:.4f}")
                    with col2:
                        st.metric("p-value", f"{comp_trigram_h.permutation_test.p_value:.4f}")
                    with col3:
                        st.metric("Cohen's d", f"{comp_trigram_h.effect_size.cohens_d:.3f}")
                    
                    if comp_trigram_h.permutation_test.is_significant:
                        st.success("✅ Различие в энтропии триграмм статистически значимо!")
                    else:
                        st.info("⚪ Различие в энтропии триграмм незначимо")
            else:
                st.warning("Недостаточно данных для сравнения триграмм")
            
            # Interpretation
            st.divider()
            st.markdown("### 💡 Интерпретация")
            st.markdown("""
            **Ключевые выводы из сравнения N-грамм:**
            
            - **Условная энтропия** показывает, насколько предсказуемы последовательности букв у каждого автора
            - Более низкая условная энтропия = более "предсказуемый" стиль письма
            - **Hapax ratio** указывает на разнообразие буквенных сочетаний
            - **Покрытие топ-10** показывает, насколько текст "сконцентрирован" на нескольких частых n-граммах
            
            Если авторы значимо различаются по этим метрикам, это надёжный индикатор разных стилей письма.
            """)

        with tab5:
            st.subheader("🔬 Детальное сравнение метрик")

            # Side-by-side metrics
            metrics_data = {
                "Метрика": [
                    "Средняя энтропия H",
                    "Средняя Perplexity",
                    "Средняя Evenness",
                    "Alphabet Utilization",
                    "Herfindahl Index",
                    "Redundancy",
                    "Correlation r",
                    "Slope",
                    "R²",
                ],
                author1_name: [
                    f"{avg1['h']:.4f}",
                    f"{avg1['perp']:.2f}",
                    f"{avg1['even']:.4f}",
                    f"{avg1['util'] * 100:.1f}%",
                    f"{avg1['hhi']:.4f}",
                    f"{avg1['redun'] * 100:.1f}%",
                    f"{batch1.correlation:.4f}" if batch1.correlation else "—",
                    f"{batch1.correlation_slope:.6f}" if batch1.correlation_slope else "—",
                    f"{batch1.correlation_r_squared:.4f}" if batch1.correlation_r_squared else "—",
                ],
                author2_name: [
                    f"{avg2['h']:.4f}",
                    f"{avg2['perp']:.2f}",
                    f"{avg2['even']:.4f}",
                    f"{avg2['util'] * 100:.1f}%",
                    f"{avg2['hhi']:.4f}",
                    f"{avg2['redun'] * 100:.1f}%",
                    f"{batch2.correlation:.4f}" if batch2.correlation else "—",
                    f"{batch2.correlation_slope:.6f}" if batch2.correlation_slope else "—",
                    f"{batch2.correlation_r_squared:.4f}" if batch2.correlation_r_squared else "—",
                ],
            }

            df_metrics = pl.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # Rényi spectrum comparison
            st.subheader("🔢 Спектр Реньи")

            # Calculate average Rényi entropies
            renyi1 = {
                "h0": np.mean([r.enhanced.renyi_0 for _, r in batch1.results if r.enhanced]),
                "h1": avg1["h"],
                "h2": np.mean([r.enhanced.renyi_2 for _, r in batch1.results if r.enhanced]),
                "hinf": np.mean([r.enhanced.renyi_inf for _, r in batch1.results if r.enhanced]),
            }

            renyi2 = {
                "h0": np.mean([r.enhanced.renyi_0 for _, r in batch2.results if r.enhanced]),
                "h1": avg2["h"],
                "h2": np.mean([r.enhanced.renyi_2 for _, r in batch2.results if r.enhanced]),
                "hinf": np.mean([r.enhanced.renyi_inf for _, r in batch2.results if r.enhanced]),
            }

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=["H₀", "H₁", "H₂", "H∞"],
                    y=[renyi1["h0"], renyi1["h1"], renyi1["h2"], renyi1["hinf"]],
                    name=author1_name,
                    marker_color="#2E86AB",
                )
            )

            fig.add_trace(
                go.Bar(
                    x=["H₀", "H₁", "H₂", "H∞"],
                    y=[renyi2["h0"], renyi2["h1"], renyi2["h2"], renyi2["hinf"]],
                    name=author2_name,
                    marker_color="#A23B72",
                )
            )

            fig.update_layout(
                title="Сравнение спектров Реньи",
                xaxis_title="Порядок энтропии",
                yaxis_title="Энтропия (бит)",
                barmode="group",
                height=400,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )

            st.plotly_chart(fig, use_container_width=True, key=f"author_compare_renyi_{id(batch1)}_{id(batch2)}")

        with tab6:
            st.subheader("📊 Распределения метрик")

            # Entropy histograms
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{author1_name} - Распределение энтропии**")
                fig1 = create_entropy_histogram(batch1)
                fig1.update_layout(height=350)
                st.plotly_chart(fig1, use_container_width=True, key=f"author_compare_hist1_{id(batch1)}")

            with col2:
                st.markdown(f"**{author2_name} - Распределение энтропии**")
                fig2 = create_entropy_histogram(batch2)
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True, key=f"author_compare_hist2_{id(batch2)}")

        with tab7:
            st.subheader("📋 Таблицы данных")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{author1_name}** ({n1} произведений)")
                df1 = st.session_state.analyzer.batch_to_dataframe(batch1)
                st.dataframe(df1, use_container_width=True)

                csv1 = df1.write_csv()
                st.download_button(
                    f"📥 Скачать {author1_name} CSV",
                    csv1,
                    f"{author1_name}_analysis.csv",
                    "text/csv",
                    key="download1",
                )

            with col2:
                st.markdown(f"**{author2_name}** ({n2} произведений)")
                df2 = st.session_state.analyzer.batch_to_dataframe(batch2)
                st.dataframe(df2, use_container_width=True)

                csv2 = df2.write_csv()
                st.download_button(
                    f"📥 Скачать {author2_name} CSV",
                    csv2,
                    f"{author2_name}_analysis.csv",
                    "text/csv",
                    key="download2",
                )
                
        with tab8:
            st.header("Полный сравнительный отчет")
            
            st.subheader(f"👤 {author1_name}")
            with st.expander(f"Развернуть отчет по {author1_name}", expanded=False):
                # Show summary for first author
                st.markdown(f"**Всего текстов:** {len(batch1.results)}")
                if batch1.extended_stats:
                    st.metric("Средняя энтропия", f"{batch1.extended_stats.mean:.4f}")
                
                # Show first 3 texts as example
                for name, res in batch1.results[:3]:
                    st.markdown(f"**{name}**")
                    render_full_report(res)
                if len(batch1.results) > 3:
                    st.info(f"...и еще {len(batch1.results)-3} текстов")
            
            st.divider()
            
            st.subheader(f"👤 {author2_name}")
            with st.expander(f"Развернуть отчет по {author2_name}", expanded=False):
                # Show summary for second author
                st.markdown(f"**Всего текстов:** {len(batch2.results)}")
                if batch2.extended_stats:
                    st.metric("Средняя энтропия", f"{batch2.extended_stats.mean:.4f}")
                
                # Show first 3 texts as example
                for name, res in batch2.results[:3]:
                    st.markdown(f"**{name}**")
                    render_full_report(res)
                if len(batch2.results) > 3:
                    st.info(f"...и еще {len(batch2.results)-3} текстов")

