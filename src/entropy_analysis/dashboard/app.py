"""
Streamlit Dashboard for Entropy Analysis.

A beautiful, interactive web interface for text entropy analysis.
"""

import streamlit as st

from entropy_analysis.dashboard.components.sidebar import sidebar_config
from entropy_analysis.dashboard.state import init_session_state
from entropy_analysis.dashboard.tabs.about import about_page
from entropy_analysis.dashboard.tabs.batch_analysis import batch_analysis
from entropy_analysis.dashboard.tabs.comparison import author_comparison, text_comparison
from entropy_analysis.dashboard.tabs.rolling import rolling_analysis, split_analysis
from entropy_analysis.dashboard.tabs.single_analysis import single_text_analysis

# Page configuration
st.set_page_config(
    page_title="Энтропийный Анализ Текстов",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Светлая тема - черные буквы на белом фоне
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Принудительно светлый фон везде */
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Все заголовки - черные */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Весь текст - черный */
    p, span, div, label, li, td, th {
        color: #000000 !important;
    }

    /* Боковая панель - светлая */
    [data-testid="stSidebar"] {
        background-color: #F5F5F5 !important;
    }

    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* Метрики */
    .stMetric {
        background: white !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .stMetric label {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }

    /* Табы */
    .stTabs [data-baseweb="tab"] {
        background-color: #E8E8E8 !important;
        color: #000000 !important;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2E86AB !important;
        color: #FFFFFF !important;
    }

    /* Поля ввода */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
    }

    /* Кнопки */
    .stButton button {
        background-color: #2E86AB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
    }

    .stButton button:hover {
        background-color: #1B4965 !important;
    }

    /* Сообщения */
    .stSuccess {
        background-color: #D4EDDA !important;
        color: #155724 !important;
    }

    .stWarning {
        background-color: #FFF3CD !important;
        color: #856404 !important;
    }

    .stInfo {
        background-color: #D1ECF1 !important;
        color: #0C5460 !important;
    }

    /* Таблицы */
    table, thead, tbody, tr, td, th {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Загрузчик файлов */
    [data-testid="stFileUploader"] {
        background-color: #F5F5F5 !important;
    }

    [data-testid="stFileUploader"] * {
        color: #000000 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """Main application."""
    init_session_state()

    # Header
    st.title("📊 Энтропийный Анализ Текстов")
    st.caption("Комплексный инструмент для статистического анализа текстов")

    # Sidebar
    options = sidebar_config()

    # Main tabs
    tabs = st.tabs(
        [
            "📄 Один текст",
            "👥 Сравнение авторов",
            "✂️ С разделителями",
            "📁 Пакетный",
            "🔄 Сравнение текстов",
            "📈 Динамика",
            "ℹ️ О проекте",
        ]
    )

    with tabs[0]:
        single_text_analysis(options)

    with tabs[1]:
        author_comparison(options)

    with tabs[2]:
        split_analysis(options)

    with tabs[3]:
        batch_analysis(options)

    with tabs[4]:
        text_comparison(options)

    with tabs[5]:
        rolling_analysis(options)

    with tabs[6]:
        about_page()


if __name__ == "__main__":
    main()
