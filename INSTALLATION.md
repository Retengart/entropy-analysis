# Руководство по установке и запуску

## Быстрый старт с Docker

### Вариант 1: Только веб-дашборд
```bash
docker run -p 8501:8501 entropy-analysis:latest
```

Откройте в браузере: http://localhost:8501

### Вариант 2: Дашборд + API (docker-compose)
```bash
docker-compose up -d
```

**Сервисы:**
- Streamlit Dashboard: http://localhost:8501
- FastAPI (API + документация): http://localhost:8000/docs

### Остановка контейнеров
```bash
docker-compose down
```

## Локальная установка (без Docker)

### Требования
- Python 3.11 или выше
- uv (рекомендуется) или pip

### Установка через uv (рекомендуется)

1. Установите uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Установите зависимости:
```bash
cd entropy-analysis-py
uv sync
```

3. Запустите веб-дашборд:
```bash
uv run entropy-analysis dashboard
```

Или API:
```bash
uv run entropy-analysis serve
```

### Установка через pip

```bash
pip install -e .
```

## Использование

### Web UI (Streamlit Dashboard)

```bash
# Локально
uv run entropy-analysis dashboard --port 8501

# Docker
docker run -p 8501:8501 entropy-analysis:latest
```

**Возможности:**
- ✅ Анализ одного текста с интерактивными графиками
- ✅ Пакетный анализ множества файлов
- ✅ Сравнение двух текстов
- ✅ Скользящая энтропия
- ✅ Настройка алфавита (rus29/rus33/custom)
- ✅ Bootstrap confidence intervals

### REST API (FastAPI)

```bash
# Локально
uv run entropy-analysis serve --port 8000

# Docker
docker run -p 8000:8000 entropy-analysis:latest \
    uvicorn entropy_analysis.api.main:app --host 0.0.0.0 --port 8000
```

**Документация:** http://localhost:8000/docs

**Endpoints:**
- `POST /api/v1/analyze` - анализ текста
- `POST /api/v1/analyze/file` - анализ файла
- `POST /api/v1/compare` - сравнение текстов
- `POST /api/v1/batch` - пакетный анализ
- `POST /api/v1/rolling` - скользящая энтропия

### CLI

```bash
# Анализ одного файла
uv run entropy-analysis analyze text.txt

# С подробным выводом
uv run entropy-analysis analyze text.txt --verbose --bootstrap

# Вывод в JSON
uv run entropy-analysis analyze text.txt --json result.json

# Пакетный анализ
uv run entropy-analysis batch ./texts/ --csv results.csv

# Сравнение текстов
uv run entropy-analysis compare text1.txt text2.txt

# Справка
uv run entropy-analysis --help
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

print(f"Энтропия: {result.shannon_entropy:.4f} бит")
print(f"95% CI: [{result.bootstrap.ci_lower:.4f}, {result.bootstrap.ci_upper:.4f}]")

# Пакетный анализ
texts = [
    ("pushkin.txt", pushkin_text),
    ("lermontov.txt", lermontov_text),
]
batch = analyzer.analyze_batch(texts)

print(f"Корреляция H vs N: {batch.correlation:.4f}")

# Сравнение текстов
comparison = analyzer.compare(text1, text2, "Пушкин", "Лермонтов")
print(f"JS Divergence: {comparison.js_divergence:.4f}")

# Визуализация
from entropy_analysis.visualization import create_letter_distribution_chart

fig = create_letter_distribution_chart(result, sort_by="frequency")
fig.show()
```

## Тестирование

```bash
# Запуск всех тестов
uv run pytest

# С покрытием кода
uv run pytest --cov=src/entropy_analysis

# Конкретный файл
uv run pytest tests/test_core.py -v
```

## Примеры использования API

### cURL

```bash
# Анализ текста
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Мой дядя самых честных правил"}'

# Сравнение текстов
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "Текст первый...",
    "text2": "Текст второй...",
    "name1": "Пушкин",
    "name2": "Лермонтов"
  }'
```

### Python requests

```python
import requests

# Анализ
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={
        "text": "Мой дядя самых честных правил",
        "include_bootstrap": True,
        "bootstrap_iterations": 1000
    }
)

result = response.json()
print(f"H = {result['shannon_entropy']:.4f}")
```

### JavaScript/fetch

```javascript
const response = await fetch('http://localhost:8000/api/v1/analyze', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Мой дядя самых честных правил',
    include_bootstrap: false
  })
});

const result = await response.json();
console.log(`H = ${result.shannon_entropy.toFixed(4)}`);
```

## Структура проекта

```
entropy-analysis-py/
├── Dockerfile              # Docker образ
├── docker-compose.yml      # Оркестрация контейнеров
├── pyproject.toml          # Зависимости (uv/pip)
├── README.md               # Основная документация
├── INSTALLATION.md         # Это файл
│
├── src/entropy_analysis/
│   ├── __init__.py         # Публичное API
│   ├── core/               # Ядро анализа
│   │   ├── analyze.py      # TextAnalyzer (главный класс)
│   │   ├── normalize.py    # Нормализация текста
│   │   ├── stats.py        # Статистика (энтропия, bootstrap)
│   │   └── metrics.py      # Метрики (KL, JS, Simpson, Zipf)
│   ├── models/             # Pydantic схемы
│   │   └── schemas.py
│   ├── visualization/      # Plotly графики
│   │   └── charts.py
│   ├── api/                # FastAPI REST API
│   │   └── main.py
│   ├── dashboard/          # Streamlit веб-дашборд
│   │   └── app.py
│   └── cli/                # CLI интерфейс
│       └── main.py
│
└── tests/                  # Тесты (26 passed ✅)
    └── test_core.py
```

## Устранение проблем

### Порт занят

```bash
# Проверить занятые порты
sudo lsof -i :8501
sudo lsof -i :8000

# Использовать другой порт
uv run entropy-analysis dashboard --port 8502
```

### Ошибка импорта модулей

```bash
# Переустановить зависимости
uv sync --reinstall
```

### Docker: контейнер не запускается

```bash
# Просмотр логов
docker logs entropy-dashboard
docker logs entropy-api

# Пересобрать образ
docker-compose build --no-cache
```

## Производительность

- **CLI анализ**: ~1000 текстов/сек
- **API throughput**: ~500 req/sec
- **Memory**: ~200MB (базовый), до 1GB (с bootstrap)

## Обновление

```bash
# Через uv
cd entropy-analysis-py
git pull
uv sync

# Docker
docker-compose pull
docker-compose up -d
```

