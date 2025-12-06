"""
Sidebar configuration component.
"""

import streamlit as st

from entropy_analysis.core.analyze import TextAnalyzer
from entropy_analysis.core.normalize import Alphabet


def sidebar_config():
    """Render sidebar configuration."""
    st.sidebar.title("⚙️ Настройки")

    st.sidebar.subheader("Алфавит")
    alphabet_choice = st.sidebar.selectbox(
        "Выберите алфавит",
        options=["rus29", "rus33", "custom"],
        index=0,
        help="rus29: без ё, й, ъ, ь (29 букв). rus33: полный алфавит (33 буквы)",
    )

    custom_letters = None
    if alphabet_choice == "custom":
        custom_letters = st.sidebar.text_input(
            "Пользовательский алфавит", value="абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        )

    keep_yo = st.sidebar.checkbox("Сохранять ё", value=False)
    keep_j = st.sidebar.checkbox("Сохранять й", value=False)
    min_token_len = st.sidebar.slider("Мин. длина слова", 1, 5, 1)

    # Update analyzer if settings changed
    current_config = (alphabet_choice, custom_letters, keep_yo, keep_j, min_token_len)

    if "last_config" not in st.session_state or st.session_state.last_config != current_config:
        st.session_state.analyzer = TextAnalyzer.create(
            alphabet=Alphabet(alphabet_choice),
            custom_letters=custom_letters,
            keep_yo=keep_yo,
            keep_j=keep_j,
            min_token_len=min_token_len,
        )
        st.session_state.last_config = current_config
        
    # Reset button
    if st.sidebar.button("🗑️ Очистить историю"):
        st.session_state.analysis_results = []
        st.rerun()

    st.sidebar.info(
        "💡 **Совет:** Чтобы прервать долгий анализ, используйте кнопку 'Stop' (⏹️) в правом верхнем углу браузера."
    )

    st.sidebar.divider()

    st.sidebar.subheader("Опции анализа")
    include_bootstrap = st.sidebar.checkbox("Bootstrap CI", value=False)
    bootstrap_n = (
        st.sidebar.slider("Bootstrap итерации", 100, 5000, 1000) if include_bootstrap else 1000
    )
    include_complexity = st.sidebar.checkbox("Анализ сжимаемости", value=False)
    include_advanced_metrics = st.sidebar.checkbox(
        "Продвинутые метрики (медленно)",
        value=False,
        help="Включает Hurst Exponent, Zipf-Mandelbrot, Pierrehumbert β. Вычисление может занять время.",
    )
    show_verbose_logs = st.sidebar.checkbox(
        "Подробный вывод",
        value=False,
        help="Показывать детальный лог выполнения операций",
    )

    return {
        "include_bootstrap": include_bootstrap,
        "bootstrap_n": bootstrap_n,
        "include_complexity": include_complexity,
        "include_advanced_metrics": include_advanced_metrics,
        "show_verbose_logs": show_verbose_logs,
    }



