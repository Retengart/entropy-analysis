# Entropy Analysis 📊

Комплексный инструмент для энтропийного анализа текстов на Python с интерактивной визуализацией.

## Возможности

### 📊 Базовые метрики
- **Энтропия Шеннона** — классическая мера информационной энтропии
- **Коррекция Миллера-Мэдоу** — компенсация смещения на малых выборках
- **Bootstrap анализ** — доверительные интервалы методом бутстрэпа
- **KL/JS Divergence** — сравнение распределений между текстами
- **Индексы разнообразия** — Simpson, Gini-Simpson
- **Анализ закона Ципфа** — проверка соответствия распределения
- **Скользящая энтропия** — анализ динамики внутри текста

### 🔬 Улучшенные метрики (НОВОЕ!)
- **Perplexity** — 2^H, эффективное число категорий в распределении
- **Alphabet Utilization** — процент использования алфавита
- **Evenness (Pielou's J)** — равномерность распределения
- **Herfindahl Index** — концентрация распределения
- **Спектр Реньи** — H₀, H₁, H₂, H∞ (4 обобщённые энтропии)
- **Uniformity Distance** — отклонение от равномерного распределения
- **Redundancy** — избыточность текста

### 📊 Статистические тесты (НОВОЕ!)
- **Permutation Test** — самый строгий тест, без предположений о распределении
- **Effect Size (Cohen's d)** — насколько БОЛЬШОЕ различие (не только значимое)
- **Bootstrap CI для различий** — доверительный интервал для разницы метрик
- **Информационные расстояния** — Wasserstein, Hellinger, Bhattacharyya
- **Комплексное сравнение** — все тесты сразу для максимальной объективности

### ✂️ Split-анализ (НОВОЕ!)
- **Анализ с разделителями** — автоматическое разделение текста по маркерам (`***`)
- **График H vs N** — зависимость энтропии от количества слов
- **Линейная регрессия** — уравнение, R², p-value
- **Корреляция** — статистический анализ связи H и N

### 👥 Сравнение авторов (НОВОЕ!)
- **Двух-авторский анализ** — сравнение больших сборников (сотни текстов)
- **Паттерны авторов** — различия в корреляциях H vs N
- **Сравнительные графики** — визуализация различий стилей
- **Детальная статистика** — 10+ метрик для каждого автора

### 🎨 Интерфейсы
- **Интерактивные графики** — Plotly для веб-визуализации
- **REST API** — FastAPI бэкенд
- **Веб-дашборд** — Streamlit интерфейс с 7 вкладками
- **CLI** — командная строка для быстрого анализа

## Установка

```bash
# Клонировать проект
cd entropy-analysis-py

# Установить через uv (рекомендуется)
uv sync

# Или через pip
pip install -e .
```

## Быстрый старт

### CLI

```bash
# Анализ одного файла
entropy-analysis analyze text.txt

# Анализ с подробным выводом
entropy-analysis analyze text.txt --verbose --bootstrap

# Пакетный анализ
entropy-analysis batch ./texts/ --csv results.csv

# Сравнение двух текстов
entropy-analysis compare pushkin.txt lermontov.txt

# НОВОЕ: Split-анализ (текст с разделителями ***)
entropy-analysis split-analyze poems.txt --show --html chart.html

# Запуск веб-дашборда (рекомендуется!)
entropy-analysis dashboard

# Запуск API сервера
entropy-analysis serve --port 8000
```

### Python API

```python
from entropy_analysis import TextAnalyzer

# Создать анализатор
analyzer = TextAnalyzer.create(alphabet="rus29")

# Анализ текста
result = analyzer.analyze(
    "Мой дядя самых честных правил...",
    include_bootstrap=True
)

print(f"Entropy: {result.shannon_entropy:.4f} bits")
print(f"95% CI: [{result.bootstrap.ci_lower:.4f}, {result.bootstrap.ci_upper:.4f}]")

# НОВОЕ: Улучшенные метрики
if result.enhanced:
    print(f"Perplexity: {result.enhanced.perplexity:.2f}")
    print(f"Evenness: {result.enhanced.evenness:.3f}")
    print(f"Alphabet Utilization: {result.enhanced.alphabet_utilization*100:.1f}%")

# Сравнение текстов
comparison = analyzer.compare(text1, text2)
print(f"JS Divergence: {comparison.js_divergence:.4f}")

# НОВОЕ: Split-анализ (текст с разделителями)
text = """Стих 1
***
Стих 2
***
Стих 3"""

batch = analyzer.split_and_analyze(text, delimiter="***")
print(f"Сегментов: {len(batch.results)}")
print(f"Корреляция H vs N: r = {batch.correlation:.4f}")
print(f"Slope: {batch.correlation_slope:.6f}")
```

### Визуализация

```python
from entropy_analysis.visualization import (
    create_letter_distribution_chart,
    create_correlation_scatter,
)

# Интерактивный график распределения букв
fig = create_letter_distribution_chart(result, sort_by="frequency")
fig.show()

# График корреляции для пакетного анализа
batch = analyzer.analyze_directory("./texts/")
fig = create_correlation_scatter(batch, label_top_n=5)
fig.show()
```

## API Endpoints

После запуска `entropy-analysis serve`:

- `POST /api/v1/analyze` — анализ текста
- `POST /api/v1/analyze/file` — анализ загруженного файла
- `POST /api/v1/compare` — сравнение двух текстов
- `POST /api/v1/batch` — пакетный анализ
- `POST /api/v1/rolling` — скользящая энтропия
- `POST /api/v1/recognition` — вероятностное распознавание (лаба 40) по таблицам P(A_i) и P(x|A)
- `POST /api/v1/recognition/batch` — обучение таблиц P(A_i), P(x|A) на размеченных сегментах и предсказания
- `GET /docs` — Swagger документация

## Метрики

### Базовые метрики

| Метрика | Описание |
|---------|----------|
| H (Shannon Entropy) | Информационная энтропия распределения **всех букв текста** (не только первых букв слов). Рассчитывается по частотам всех букв в тексте (бит) |
| H_norm | Нормализованная энтропия: H / log₂(\|Σ\|) |
| H_MM | Коррекция Миллера-Мэдоу для малых выборок |
| x̄ (mean rank) | Средний ранг буквы в алфавите |
| σ (std rank) | Стандартное отклонение ранга |
| Simpson Index | Вероятность совпадения двух случайных букв |
| Gini-Simpson | 1 - Simpson (мера разнообразия) |
| Zipf α | Экспонента закона Ципфа |
| KL Divergence | Расстояние Кульбака-Лейблера между распределениями |
| JS Divergence | Симметричное расстояние Йенсена-Шеннона |

### 🔬 Улучшенные метрики (НОВОЕ!)

| Метрика | Описание | Диапазон |
|---------|----------|----------|
| **Perplexity** | 2^H — эффективное число категорий в распределении | [1, ∞) |
| **Alphabet Utilization** | Процент использованных букв алфавита | [0%, 100%] |
| **Evenness (Pielou's J)** | Равномерность распределения | [0, 1] |
| **Herfindahl Index** | Концентрация распределения | [1/n, 1] |
| **Uniformity Distance** | Расстояние от равномерного распределения | [0, 1] |
| **Uniqueness Ratio** | Уникальных букв на слово | [0, ∞) |
| **Redundancy** | Избыточность (1 - H/H_max) | [0%, 100%] |
| **H₀ (Hartley)** | log₂(число используемых букв) | - |
| **H₂ (Collision)** | Энтропия столкновений | - |
| **H∞ (Min-entropy)** | Энтропия наихудшего случая | - |

**Подробнее:** см. `ENHANCED_METRICS_GUIDE.md`

## 👥 Сравнение авторов (НОВОЕ!)

Для сравнения двух больших сборников (например, Пушкин vs Лермонтов):

### Веб-интерфейс (рекомендуется!)

1. Запустите dashboard: `uv run entropy-analysis dashboard`
2. Перейдите на вкладку **"👥 Сравнение авторов"**
3. Загрузите два файла с произведениями, разделёнными `***`
4. Нажмите "Сравнить авторов"

**Что вы увидите:**
- 📈 **Корреляция H vs N** - паттерны авторов (ГЛАВНАЯ метрика!)
- 🔢 **Perplexity** - эффективное разнообразие (2^H)
- 🎯 **Evenness** - равномерность стиля
- 📊 **Спектр Реньи** - детальные характеристики
- 📉 **Сравнительные графики** - визуализация различий

### Эффективность метода

**Научная валидация** (Пушкин vs Лермонтов, 70 текстов):

- ✅ **Entropy-based method: 71.43%** accuracy
- ⚪ TF-IDF + SVM baseline: 47.62% accuracy
- 📊 **Улучшение: +23.81%** (1.50x лучше baseline)

**Ключевые метрики для атрибуции:**
1. **Perplexity (2^H)** — эффективное число категорий
2. **Evenness (Pielou's J)** — равномерность распределения
3. **Rényi spectrum** — H₀, H₂, H∞ для детального анализа
4. **Alphabet utilization** — процент использованного алфавита
5. **Correlation H vs N** — паттерн изменения энтропии от длины

**Полные результаты:** см. `benchmarks/results/`  
**Методология:** см. `benchmarks/README.md`

### 🧭 Алгоритм распознавания (лаба 40)
- Новая вкладка `🧭 Распознавание` в Streamlit-дэшборде строит информативность признаков, ошибки P(e) и лучшую пару признаков. Поддерживает опциональный шум.
- Новая вкладка `🧭 Пакетное распознавание` обучает таблицы вероятностей по размеченным сегментам (например, стихи с разделителями/метками) и сразу предсказывает классы по каждому сегменту.
- REST: `POST /api/v1/recognition`

Пример запроса:
```json
{
  "classes": ["C1", "C2", "C3"],
  "priors": [0.3, 0.5, 0.2],
  "features": [
    {"name": "f1", "values_count": 2, "conditional": [[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]]},
    {"name": "f2", "values_count": 3, "conditional": [[0.2, 0.5, 0.3], [0.3, 0.4, 0.3], [0.1, 0.6, 0.3]]}
  ],
  "error_target": 0.05,
  "noise": {"factor": 0.0, "mode": "uniform", "seed": 123, "renormalize": true}
}
```

Пакетное распознавание (обучение + предсказания):
```json
{
  "segments": [
    {"name": "A_1", "label": "author1", "text": "Текст автора 1"},
    {"name": "B_1", "label": "author2", "text": "Текст автора 2"}
  ],
  "error_target": 0.05,
  "smoothing": 0.001,
  "noise": {"factor": 0.0, "mode": "uniform", "seed": 123, "renormalize": true}
}
```

## Алфавиты

- **rus29** — 29 букв: ё→е, й→и (замена), ъ и ь исключаются — стандартная методика
- **rus33** — полный алфавит из 33 букв (все буквы сохраняются)
- **custom** — пользовательский набор символов

## Технологический стек

- **uv** — быстрый менеджер пакетов
- **Polars** — высокопроизводительная работа с данными
- **NumPy/SciPy** — численные вычисления
- **Plotly** — интерактивная визуализация
- **FastAPI** — REST API
- **Streamlit** — веб-дашборд
- **Typer/Rich** — CLI интерфейс

## Структура проекта

```
entropy-analysis-py/
├── pyproject.toml          # Зависимости и конфигурация
├── README.md
└── src/
    └── entropy_analysis/
        ├── __init__.py
        ├── core/               # Ядро анализа
        │   ├── analyze.py      # Главный анализатор
        │   ├── normalize.py    # Нормализация текста
        │   ├── stats.py        # Статистические функции
        │   └── metrics.py      # Продвинутые метрики
        ├── models/             # Pydantic схемы
        │   └── schemas.py
        ├── visualization/      # Plotly графики
        │   └── charts.py
        ├── api/                # FastAPI бэкенд
        │   └── main.py
        ├── dashboard/          # Streamlit UI
        │   └── app.py
        └── cli/                # CLI интерфейс
            └── main.py
```

## Теоретическая основа

Энтропия Шеннона для распределения начальных букв:

$$H = -\sum_{i} p_i \log_2 p_i$$

где $p_i = n_i / N$ — эмпирическая частота $i$-й буквы.

Средний ранг и стандартное отклонение:

$$\bar{x} = \sum_i r_i \cdot p_i$$

$$\sigma = \sqrt{\sum_i (r_i - \bar{x})^2 \cdot p_i}$$

## Лицензия

MIT

## Благодарности

Проект основан на методике энтропийного анализа поэтических текстов.
Переписан с Rust на Python для расширенной визуализации и аналитики.

