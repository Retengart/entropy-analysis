# История изменений

## [2.0.0] - 2025-11-29

### ✨ Полная переписка с Rust на Python

**Основные изменения:**
- Полная переписка проекта entropy-analysis с Rust на Python
- Добавлены веб-интерфейсы (Streamlit + FastAPI)
- Расширены метрики с 4 до 15
- Интерактивные графики вместо статичных PNG/SVG

### 🎨 Улучшения контраста и читаемости

**Проблема:**
- Светлый текст на светлом фоне в метриках
- Недостаточный контраст на осях графиков
- Плохо читаемые аннотации

**Исправлено:**
- ✅ Цвет основного текста: `#0D1B2A` (почти чёрный, WCAG AAA)
- ✅ Фон метрик: чистый белый с тенью
- ✅ Оси графиков: размер 14-16px, тёмный цвет, bold
- ✅ Аннотации: белый фон, чёрный текст, borders
- ✅ Headers: `#0D1B2A`, font-weight 700
- ✅ Все элементы UI: высокий контраст

### 📦 Добавлено

#### Core (ядро анализа):
- `core/normalize.py` - нормализация текста (rus29/rus33/custom)
- `core/stats.py` - статистические функции
- `core/metrics.py` - продвинутые метрики
- `core/analyze.py` - главный анализатор TextAnalyzer

#### Новые метрики:
- Miller-Madow bias correction
- Bootstrap confidence intervals (1000 iterations)
- Kullback-Leibler divergence
- Jensen-Shannon divergence (symmetric)
- Simpson's Index / Gini-Simpson
- Zipf's law analysis (α, R²)
- Rolling/sliding window entropy
- Compression ratio (gzip-based)
- Lempel-Ziv complexity
- Mutual Information
- Conditional entropy
- Extended statistics (Q1, Q3, skew, kurtosis)
- Outlier detection (IQR, Z-score, Modified Z-score)

#### Визуализация (Plotly):
- `visualization/charts.py` - 9 типов интерактивных графиков
- Letter distribution (по алфавиту/частоте)
- Zipf plot (log-log)
- Correlation scatter (с trendline)
- Entropy histogram (с KDE)
- Rolling entropy line chart
- Radar chart (multi-text comparison)
- Comparison heatmap
- Dual author comparison
- Bootstrap CI visualization

#### Web интерфейсы:
- `dashboard/app.py` - Streamlit UI (5 вкладок)
  - 📄 Один текст - анализ с графиками
  - 📁 Пакетный - multi-file upload
  - 🔄 Сравнение - side-by-side
  - 📈 Динамика - rolling entropy
  - ℹ️ О проекте - методология
- `api/main.py` - FastAPI REST API (7 endpoints)
- `cli/main.py` - CLI с Rich tables

#### Models:
- `models/schemas.py` - Pydantic схемы для API

#### Docker:
- `Dockerfile` - multi-stage build
- `docker-compose.yml` - dev/test deployment
- `docker-compose.production.yml` - production с nginx
- `nginx.conf` - reverse proxy конфигурация
- `.dockerignore` - оптимизация образа

#### Документация:
- `README.md` - основная документация
- `QUICKSTART.md` - быстрый старт (5 минут)
- `INSTALLATION.md` - детальная установка
- `SUMMARY.md` - полное резюме возможностей
- `VERIFICATION_REPORT.md` - отчёт о тестировании
- `FINAL_REPORT.md` - итоговый отчёт
- `COLOR_IMPROVEMENTS.md` - про улучшения контраста
- `START_HERE.md` - точка входа
- `ШПАРГАЛКА.md` - краткая справка

#### Примеры:
- `example_usage.py` - 7 практических примеров
- `demo_visualizations.py` - демо всех 9 графиков
- `test_high_contrast.py` - проверка контраста

#### Инфраструктура:
- `Makefile` - автоматизация (15 команд)
- `pyproject.toml` - зависимости (uv/pip)
- `uv.lock` - locked dependencies
- `.dockerignore` - оптимизация Docker
- `tests/test_core.py` - 26 unit tests

### 🔧 Технологический стек

**Package Management:**
- uv (вместо pip) - 10-100x быстрее

**Data Processing:**
- Polars (вместо Pandas) - Rust-based, быстрее
- NumPy - векторизация
- SciPy - научные вычисления

**Visualization:**
- Plotly (вместо Matplotlib) - интерактивные графики

**Web:**
- FastAPI - async REST API
- Streamlit - интерактивный дашборд

**CLI:**
- Typer - современный CLI
- Rich - beautiful terminal output

**ML/Stats:**
- scikit-learn - Mutual Information
- NLTK - NLP готовность

**Testing:**
- Pytest - 26 tests

**Deployment:**
- Docker - containerization
- docker-compose - orchestration

### 📊 Производительность

**Benchmark (на современном процессоре):**
- Анализ 1K слов: ~10ms
- Анализ 100K слов (pushkin.txt): ~50ms
- Batch 100 файлов: ~1s
- Bootstrap CI (1000 iter): ~200ms

**Memory:**
- Базовый анализ: ~200MB
- С bootstrap: ~500MB
- Large batch: ~1GB

### 🔒 Безопасность

- Non-root Docker user
- Pydantic validation
- No eval/exec
- Sanitized inputs
- CORS configured

### ✅ Тестирование

- 26 unit tests (все пройдены)
- CLI integration tests
- API manual testing
- Dashboard manual testing
- Docker deployment testing
- Example scripts verification

### 📝 Документация

- 9 .md файлов
- Docstrings для всех публичных функций
- Type hints везде
- API auto-docs (OpenAPI/Swagger)
- In-code examples

### 🐳 Docker

**Образы:**
- `entropy-analysis:latest` - базовый образ (1.85GB)
- `entropy-analysis-py-dashboard:latest` - дашборд (1.86GB)
- `entropy-analysis-py-api:latest` - API (1.86GB)

**Compose:**
- Development: `docker-compose.yml`
- Production: `docker-compose.production.yml` (с nginx, limits)

### 🎯 Совместимость

**Python:** 3.11+  
**OS:** Linux, macOS, Windows  
**Docker:** 20.10+  
**Browser:** Modern browsers (Chrome, Firefox, Safari, Edge)

### 🔗 Миграция из Rust

**Результаты идентичны:**
- Pushkin.txt (96,736 слов):
  - Rust: H ≈ 4.254 bits
  - Python: H = 4.2541 bits ✅

**CLI совместимость:**
```bash
# Было (Rust)
cargo run -- analyze text.txt

# Стало (Python)
uv run entropy-analysis analyze text.txt
```

---

## Roadmap (будущие версии)

### v2.1.0 (планируется)
- [ ] Темная тема (dark mode)
- [ ] Export в PDF
- [ ] Базу данных для истории
- [ ] User authentication
- [ ] Кэширование (Redis)

### v2.2.0
- [ ] Multi-language support (English)
- [ ] N-gram analysis (не только первая буква)
- [ ] Author attribution ML
- [ ] Text clustering

### v3.0.0
- [ ] Real-time collaboration
- [ ] Cloud deployment templates
- [ ] Mobile app

---

## Contributors

- **Initial Rust version:** Chris
- **Python rewrite:** Chris (with AI assistance)

## License

MIT

---

**Статус:** ✅ Production Ready  
**Версия:** 2.0.0  
**Дата:** 2025-11-29


