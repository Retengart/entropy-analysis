"""
Metric rendering components.
"""

import plotly.graph_objects as go
import streamlit as st

from entropy_analysis.visualization.charts import (
    create_letter_distribution_chart,
    create_zipf_plot,
)


def render_metrics(result):
    """Render key metrics in a nice layout."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📝 Слов", value=f"{result.n_words:,}", help="Общее количество слов в тексте"
        )

    with col2:
        h_value = f"{result.shannon_entropy:.4f}" if result.shannon_entropy else "—"
        st.metric(
            label="📊 Энтропия H", value=h_value, help="Информационная энтропия Шеннона (бит)"
        )

    with col3:
        mean_value = f"{result.mean_rank:.2f}" if result.mean_rank else "—"
        st.metric(
            label="📈 Средний ранг", value=mean_value, help="Математическое ожидание ранга буквы"
        )

    with col4:
        sigma_value = f"{result.std_rank:.2f}" if result.std_rank else "—"
        st.metric(label="📉 σ (ранг)", value=sigma_value, help="Стандартное отклонение ранга")


def render_bootstrap_info(result):
    """Render bootstrap confidence interval."""
    if result.bootstrap:
        st.subheader("🎯 Доверительный интервал (Bootstrap)")

        b = result.bootstrap
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Оценка H", f"{b.estimate:.4f}")
        with col2:
            st.metric("Стд. ошибка", f"±{b.std_error:.4f}")
        with col3:
            st.metric(
                f"{b.confidence_level * 100:.0f}% CI", f"[{b.ci_lower:.4f}, {b.ci_upper:.4f}]"
            )


def render_advanced_metrics(result, key_suffix: str = ""):
    """Render advanced metrics."""
    st.subheader("📊 Расширенные метрики")

    col1, col2, col3 = st.columns(3)

    with col1:
        if result.normalized_entropy:
            st.metric("H нормализованная", f"{result.normalized_entropy.h_normalized:.4f}")
            st.metric("Эффективность", f"{result.normalized_entropy.efficiency * 100:.1f}%")

        if result.miller_madow_entropy:
            st.metric("H (Miller-Madow)", f"{result.miller_madow_entropy:.4f}")

    with col2:
        if result.simpson_index is not None:
            st.metric("Индекс Симпсона", f"{result.simpson_index:.4f}")
        if result.gini_simpson_index is not None:
            st.metric("Gini-Simpson", f"{result.gini_simpson_index:.4f}")

    with col3:
        if result.zipf_alpha is not None:
            st.metric("Zipf α", f"{result.zipf_alpha:.3f}")
        if result.zipf_r_squared is not None:
            st.metric("Zipf R²", f"{result.zipf_r_squared:.4f}")
        if result.compression_ratio is not None:
            st.metric("Сжимаемость", f"{result.compression_ratio:.3f}")

    # Enhanced Metrics Section
    if result.enhanced:
        st.subheader("🔬 Улучшенные метрики")

        # Tabs for better organization
        enh_tab1, enh_tab2, enh_tab3, enh_tab4 = st.tabs(
            ["📈 Интерпретация", "🎯 Распределение", "🔢 Rényi Спектр", "📚 Лексика и Структура"]
        )

        with enh_tab1:
            st.markdown("**Интерпретируемые метрики для анализа**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Perplexity (Перплексия)",
                    f"{result.enhanced.perplexity:.2f}",
                    help="Эффективное число категорий (2^H). Показывает, скольким равновероятным буквам эквивалентно распределение.",
                )
                st.caption(f"≈ {result.enhanced.perplexity:.0f} эквивалентных букв")

            with col2:
                util_pct = result.enhanced.alphabet_utilization * 100
                st.metric(
                    "Использование алфавита",
                    f"{util_pct:.1f}%",
                    help="Доля букв алфавита, которые реально используются в тексте.",
                )
                if util_pct > 80:
                    st.caption("🟢 Отлично!")
                elif util_pct > 60:
                    st.caption("🟡 Хорошо")
                else:
                    st.caption("🔴 Ограниченное")

            with col3:
                st.metric(
                    "Uniqueness Ratio",
                    f"{result.enhanced.uniqueness_ratio:.3f}",
                    help="Количество уникальных букв на одно слово.",
                )
                st.caption("Уник. букв/слово")

        with enh_tab2:
            st.markdown("**Характеристики распределения**")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Herfindahl Index (HHI)",
                    f"{result.enhanced.herfindahl_index:.4f}",
                    help="Мера концентрации. Выше = больше концентрация (меньше разнообразие).",
                )

                st.metric(
                    "Evenness (Pielou's J)",
                    f"{result.enhanced.evenness:.3f}",
                    help="Равномерность распределения. 1.0 = идеально равномерное.",
                )
                if result.enhanced.evenness > 0.9:
                    st.caption("🟢 Очень равномерное")
                elif result.enhanced.evenness > 0.7:
                    st.caption("🟡 Умеренно равномерное")
                else:
                    st.caption("🔴 Неравномерное")

            with col2:
                st.metric(
                    "Uniformity Distance",
                    f"{result.enhanced.uniformity_distance:.4f}",
                    help="Евклидово расстояние от равномерного распределения.",
                )

                red_pct = result.enhanced.redundancy * 100
                st.metric(
                    "Redundancy (Избыточность)",
                    f"{red_pct:.1f}%",
                    help="Доля неиспользованной информационной ёмкости (1 - H/H_max).",
                )
                st.caption(f"Используется {100 - red_pct:.1f}% ёмкости")

        with enh_tab3:
            st.markdown("**Семейство энтропий Реньи**")
            st.caption(
                "Разные порядки α показывают чувствительность к различным аспектам распределения"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "H₀ (Hartley)",
                    f"{result.enhanced.renyi_0:.4f} бит",
                    help="Логарифм числа используемых букв. Не зависит от частоты.",
                )
                st.caption("α = 0: только наличие/отсутствие")

                st.metric(
                    "H₁ (Shannon)",
                    f"{result.shannon_entropy:.4f} бит",
                    help="Стандартная энтропия Шеннона.",
                )
                st.caption("α = 1: оптимальный баланс")

            with col2:
                st.metric(
                    "H₂ (Collision)",
                    f"{result.enhanced.renyi_2:.4f} бит",
                    help="Энтропия столкновений. Более чувствительна к частым буквам.",
                )
                st.caption("α = 2: вероятность коллизии")

                st.metric(
                    "H∞ (Min-entropy)",
                    f"{result.enhanced.renyi_inf:.4f} бит",
                    help="Определяется самой частой буквой. Энтропия наихудшего случая.",
                )
                st.caption("α = ∞: наихудший случай")

            # Visualization of Rényi spectrum
            st.markdown("**Монотонность спектра:** H₀ ≥ H₁ ≥ H₂ ≥ H∞")

            renyi_data = {
                "α": ["0 (Hartley)", "1 (Shannon)", "2 (Collision)", "∞ (Min)"],
                "H": [
                    result.enhanced.renyi_0,
                    result.shannon_entropy,
                    result.enhanced.renyi_2,
                    result.enhanced.renyi_inf,
                ],
            }

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=renyi_data["α"],
                        y=renyi_data["H"],
                        marker_color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
                        text=[f"{h:.3f}" for h in renyi_data["H"]],
                        textposition="outside",
                    )
                ]
            )

            fig.update_layout(
                title="Спектр энтропий Реньи",
                xaxis_title="Порядок α",
                yaxis_title="Энтропия (бит)",
                height=300,
                showlegend=False,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )

            st.plotly_chart(fig, use_container_width=True, key=f"renyi_spectrum_{key_suffix}_{id(result)}")

        with enh_tab4:
            st.markdown("**Лексическое разнообразие и Структура**")

            col1, col2, col3 = st.columns(3)

            with col1:
                if result.yules_k:
                    st.metric(
                        "Yule's K",
                        f"{result.yules_k:.2f}",
                        help="Лексическая концентрация. Инвариантна к длине текста. Ниже = богаче лексика.",
                    )
                if result.mattr:
                    st.metric(
                        "MATTR",
                        f"{result.mattr:.4f}",
                        help="Moving Average Type-Token Ratio. Стабильная метрика лексического разнообразия.",
                    )
                elif result.bigram_entropy:
                    st.metric(
                        "H(X₂|X₁)",
                        f"{result.bigram_entropy:.3f}",
                        help="Условная энтропия биграмм. Насколько сложно предсказать следующую букву/слово.",
                    )

            with col2:
                if result.mtld:
                    st.metric(
                        "MTLD",
                        f"{result.mtld:.2f}",
                        help="Measure of Textual Lexical Diversity. Выше = богаче лексика (дольше держит разнообразие).",
                    )
                if result.gries_dp:
                    st.metric(
                        "Gries' DP (norm)",
                        f"{result.gries_dp:.4f}",
                        help="Deviation of Proportions. 1.0 = идеальная равномерность слов.",
                    )
                elif result.trigram_entropy:
                    st.metric(
                        "H(X₃|X₁,X₂)",
                        f"{result.trigram_entropy:.3f}",
                        help="Условная энтропия триграмм. Предсказуемость по двум предыдущим.",
                    )

            with col3:
                if result.burstiness:
                    st.metric(
                        "Burstiness (B)",
                        f"{result.burstiness:.4f}",
                        help="Взрывность слов. >0 = слова скапливаются, <0 = регулярное повторение.",
                    )
                if result.pierrehumbert_beta:
                    st.metric(
                        "Pierrehumbert β",
                        f"{result.pierrehumbert_beta:.3f}",
                        help="Параметр формы Weibull. <1: взрывные слова (content), ~1: случайные, >1: регулярные (function).",
                    )
            
            # Readability Indices Section
            if result.flesch_reading_ease is not None or result.gunning_fog_index is not None:
                st.divider()
                st.markdown("**📖 Индексы удобочитаемости**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if result.flesch_reading_ease is not None:
                        flesch = result.flesch_reading_ease
                        # Determine interpretation
                        if flesch >= 90:
                            flesch_desc = "Очень лёгкий"
                            flesch_color = "🟢"
                        elif flesch >= 80:
                            flesch_desc = "Лёгкий"
                            flesch_color = "🟢"
                        elif flesch >= 70:
                            flesch_desc = "Достаточно лёгкий"
                            flesch_color = "🟡"
                        elif flesch >= 60:
                            flesch_desc = "Средней сложности"
                            flesch_color = "🟡"
                        elif flesch >= 50:
                            flesch_desc = "Довольно сложный"
                            flesch_color = "🟠"
                        elif flesch >= 30:
                            flesch_desc = "Сложный"
                            flesch_color = "🔴"
                        else:
                            flesch_desc = "Очень сложный"
                            flesch_color = "🔴"
                        
                        st.metric(
                            "Индекс Флеша",
                            f"{flesch:.1f}",
                            delta=flesch_desc,
                            delta_color="off",
                            help="Индекс удобочитаемости Флеша. Выше = легче читать. "
                                 "90-100: очень лёгкий, 60-70: средний, 30-50: сложный, 0-30: очень сложный.",
                        )
                        st.caption(f"{flesch_color} {flesch_desc}")
                    
                    if result.avg_syllables_per_word is not None:
                        st.metric(
                            "Сред. слогов/слово",
                            f"{result.avg_syllables_per_word:.2f}",
                            help="Среднее количество слогов на слово. Больше = сложнее слова.",
                        )
                
                with col2:
                    if result.gunning_fog_index is not None:
                        fog = result.gunning_fog_index
                        # Determine interpretation
                        if fog < 6:
                            fog_desc = "Простой для всех"
                            fog_color = "🟢"
                        elif fog < 12:
                            fog_desc = "Легко читаемый"
                            fog_color = "🟢"
                        elif fog < 17:
                            fog_desc = "Средняя сложность"
                            fog_color = "🟡"
                        elif fog < 24:
                            fog_desc = "Высокая сложность"
                            fog_color = "🟠"
                        else:
                            fog_desc = "Очень сложный"
                            fog_color = "🔴"
                        
                        st.metric(
                            "Индекс туманности Ганнинга",
                            f"{fog:.1f}",
                            delta=fog_desc,
                            delta_color="off",
                            help="Gunning Fog Index. Указывает на уровень образования, необходимый для понимания. "
                                 "<6: простой, 7-12: легко читаемый, 13-17: средний, 18-24: сложный, >24: очень сложный.",
                        )
                        st.caption(f"{fog_color} {fog_desc}")
                    
                    if result.avg_sentence_length is not None:
                        st.metric(
                            "Сред. длина предложения",
                            f"{result.avg_sentence_length:.1f} слов",
                            help="Среднее количество слов в предложении. Больше = сложнее структура.",
                        )

            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if result.hurst_exponent:
                    hurst_val = result.hurst_exponent
                    hurst_desc = "Случайный (шум)"
                    if hurst_val > 0.55:
                        hurst_desc = "Персистентный (память)"
                    elif hurst_val < 0.45:
                        hurst_desc = "Анти-персистентный"
                    
                    st.metric(
                        "Hurst Exponent (H)",
                        f"{hurst_val:.3f}",
                        delta=hurst_desc,
                        delta_color="off",
                        help="Фрактальность текста (DFA). >0.5 указывает на наличие долгосрочной памяти и структуры.",
                    )

            with col2:
                if result.zipf_mandelbrot_beta:
                    st.metric(
                        "Zipf-Mandelbrot β",
                        f"{result.zipf_mandelbrot_beta:.3f}",
                        help="Параметр коррекции кривизны Мандельброта. Улучшает аппроксимацию для частых слов.",
                    )


def render_full_report(result):
    """Render a comprehensive report for a single text analysis result."""
    st.subheader(f"📑 Полный отчет: {result.source_name or 'Текст'}")
    
    # 1. Key Metrics
    st.markdown("### 1. Основные метрики")
    render_metrics(result)
    
    # 2. Bootstrap
    if result.bootstrap:
        st.markdown("### 2. Bootstrap анализ")
        render_bootstrap_info(result)
        
    # 3. Visualizations
    st.markdown("### 3. Визуализации")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Распределение букв (алфавит)**")
        fig1 = create_letter_distribution_chart(result, sort_by="alphabet", show_cumulative=True)
        source_id = result.source_name or "unnamed"
        st.plotly_chart(fig1, use_container_width=True, key=f"full_report_dist_{source_id}_{id(result)}")
    with col2:
        st.markdown("**Закон Ципфа**")
        fig2 = create_zipf_plot(result)
        source_id = result.source_name or "unnamed"
        st.plotly_chart(fig2, use_container_width=True, key=f"full_report_zipf_{source_id}_{id(result)}")
        
    # 4. Advanced Metrics
    st.markdown("### 4. Расширенные и продвинутые метрики")
    render_advanced_metrics(result, key_suffix="full_report")
    
    st.divider()
