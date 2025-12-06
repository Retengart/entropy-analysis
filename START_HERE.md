# 🚀 НАЧНИТЕ ЗДЕСЬ!

## Проект готов к использованию ✅

**Энтропийный Анализ Текстов v2.0** - современный Python-инструмент для статистического анализа текстов.

---

## ⚡ Быстрый старт (выберите один вариант)

### Вариант 1: Docker (рекомендуется) 🐳

```bash
# Запуск дашборда + API
docker-compose up -d

# Проверка
docker-compose ps

# Логи (если нужно)
docker-compose logs -f
```

**→ Откройте в браузере:**
- 📊 **Dashboard:** http://localhost:8501
- 🚀 **API Docs:** http://localhost:8000/docs

### Вариант 2: Локально с Makefile 🛠️

```bash
# Установить зависимости (один раз)
make install

# Запустить дашборд
make dashboard
```

**→ Откройте:** http://localhost:8501

### Вариант 3: Прямые команды 💻

```bash
# Установка (один раз)
uv sync

# Запуск дашборда
uv run entropy-analysis dashboard

# Или API
uv run entropy-analysis serve

# Или CLI
uv run entropy-analysis analyze файл.txt
```

---

## 📖 Документация

### Для начала работы:
1. **QUICKSTART.md** ← Начните отсюда! (5 минут)
2. **ШПАРГАЛКА.md** ← Частые команды

### Для установки:
3. **INSTALLATION.md** ← Детальные инструкции

### Для изучения:
4. **README.md** ← Полная документация
5. **ENHANCED_METRICS_GUIDE.md** ← Описание метрик
6. **STATISTICAL_TESTS_GUIDE.md** ← Статистические тесты

### Для сравнения авторов:
7. **AUTHOR_COMPARISON_GUIDE.md** ← Полное руководство
8. **ДЛЯ_СРАВНЕНИЯ_АВТОРОВ.md** ← Краткое руководство
9. **SPLIT_ANALYSIS_GUIDE.md** ← Анализ с разделителями

### История:
10. **CHANGELOG.md** ← История изменений

---

## 🎨 Что есть в проекте

### 3 интерфейса:

1. **📊 Web Dashboard (Streamlit)**
   - 6 вкладок с анализом
   - Drag-and-drop файлов
   - Интерактивные графики
   - CSV export

2. **🚀 REST API (FastAPI)**
   - 7 endpoints
   - Swagger UI
   - File upload
   - JSON/CSV export

3. **💻 CLI (Typer)**
   - 6 команд
   - Beautiful tables
   - Progress bars
   - JSON/CSV export

### 25+ метрик:

- Shannon Entropy (H)
- Miller-Madow correction
- Bootstrap CI
- KL/JS Divergence
- Simpson/Gini-Simpson Index
- Zipf's law + Zipf-Mandelbrot
- Yule's K, MTLD, MATTR, Gries' DP
- Perplexity, Evenness, Redundancy
- Rényi entropies (H₀, H₂, H∞)
- Hurst Exponent (DFA)
- Burstiness Index
- Delta метрики (Burrows, Eder, Cosine)
- Rolling entropy
- Compression ratio
- И другие...

### 9 типов графиков (Plotly):

- Letter distribution
- Zipf plot (log-log)
- Correlation scatter
- Entropy histogram
- Rolling entropy
- Radar comparison
- Heatmap
- Dual author comparison
- Bootstrap CI

**Все графики интерактивные!** Zoom, pan, hover.

---

## 🧪 Проверка установки

### Быстрая проверка:

```bash
# Тесты (должно быть 30/30 passed)
make test

# Примеры
make example

# Версия
uv run entropy-analysis version
```

### Полная проверка:

```bash
# Все 9 графиков (создаёт HTML файлы)
make demo-viz

# Проверка Docker
docker-compose up -d
docker-compose ps  # Должно быть 2 контейнера HEALTHY
```

---

## 🎯 Частые задачи

### Анализировать файл:

```bash
uv run entropy-analysis analyze my_text.txt --verbose
```

### Пакетный анализ:

```bash
uv run entropy-analysis batch texts_folder/ --csv results.csv
```

### Сравнить два текста:

```bash
uv run entropy-analysis compare text1.txt text2.txt
```

### Использовать в Python скрипте:

```python
from entropy_analysis import TextAnalyzer

analyzer = TextAnalyzer.create()
result = analyzer.analyze("Ваш текст здесь")

print(f"Энтропия: {result.shannon_entropy:.4f} бит")
print(f"Perplexity: {result.enhanced.perplexity:.2f}")
```

---

## 🆘 Нужна помощь?

### Проблемы с запуском?

1. **Порт занят:**
   ```bash
   uv run entropy-analysis dashboard --port 8502
   ```

2. **Модули не найдены:**
   ```bash
   uv sync --reinstall
   ```

3. **Docker не работает:**
   ```bash
   docker-compose logs dashboard
   docker-compose build --no-cache
   docker-compose up
   ```

### Где искать информацию?

- **QUICKSTART.md** - быстрый старт
- **ШПАРГАЛКА.md** - частые команды
- **README.md** - полная документация
- **example_usage.py** - примеры кода

---

## 🎉 Приятного использования!

**Проект полностью функционален и протестирован.**

**Самый простой способ начать:**

```bash
docker-compose up -d
```

Затем откройте http://localhost:8501 и начните анализировать!

---

**Версия:** 2.0.0  
**Статус:** ✅ Production Ready  
**Тесты:** ✅ 30/30 passed
