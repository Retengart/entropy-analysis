# Быстрый старт 🚀

## Самый простой способ - Docker

### 1. Запуск веб-дашборда (один контейнер)

```bash
# Собрать образ
docker build -t entropy-analysis:latest .

# Запустить дашборд
docker run -p 8501:8501 entropy-analysis:latest
```

**Откройте в браузере:** http://localhost:8501

### 2. Запуск через docker-compose (дашборд + API)

```bash
# Запуск
docker-compose up -d

# Проверка статуса
docker-compose ps

# Логи
docker-compose logs -f

# Остановка
docker-compose down
```

**URLs:**
- 📊 Веб-дашборд: http://localhost:8501
- 🚀 API документация: http://localhost:8000/docs
- ❤️ API health: http://localhost:8000/health

### 3. Production deployment

```bash
docker-compose -f docker-compose.production.yml up -d
```

## Локальная разработка (без Docker)

### Установка

```bash
# С помощью uv (быстро!)
uv sync

# Или через pip
pip install -e .
```

### Использование через Makefile

```bash
# Показать все команды
make help

# Запустить тесты
make test

# Запустить веб-дашборд
make dashboard

# Запустить API
make api

# Примеры использования
make example

# Анализ из старого проекта
make analyze-pushkin
make analyze-batch
```

### Прямые команды

```bash
# 1. Веб-дашборд (самый простой вариант)
uv run entropy-analysis dashboard

# 2. API сервер
uv run entropy-analysis serve --port 8000

# 3. CLI анализ
uv run entropy-analysis analyze ../entropy-analysis/text/pushkin.txt

# 4. Примеры на Python
uv run python example_usage.py
```

## Использование веб-дашборда

### Вкладка "📄 Один текст"

1. Выберите способ ввода:
   - Вставить текст вручную
   - Загрузить файл (.txt, .md)

2. Опционально настройте в Sidebar:
   - Алфавит (rus29/rus33/custom)
   - Сохранять ё, й
   - Bootstrap CI
   - Анализ сжимаемости

3. Нажмите "🔍 Анализировать"

4. Изучите результаты:
   - **Метрики**: N, H, x̄, σ
   - **График "Распределение"**: столбцы по алфавиту
   - **График "По частоте"**: отсортировано по вероятности
   - **График "Закон Ципфа"**: лог-лог
   - **Таблица**: детальные данные

### Вкладка "📁 Пакетный"

1. Загрузите несколько файлов (drag-and-drop или Browse)
2. Нажмите "🔍 Анализировать все"
3. Смотрите:
   - Гистограмму энтропии
   - Корреляцию H vs N
   - Таблицу результатов
   - Скачайте CSV

### Вкладка "🔄 Сравнение"

1. Введите два текста
2. Нажмите "🔍 Сравнить"
3. Получите:
   - KL divergence, JS divergence
   - Cosine similarity
   - Side-by-side графики
   - Радарное сравнение

### Вкладка "📈 Динамика"

1. Введите длинный текст
2. Настройте размер окна и шаг
3. Нажмите "🔍 Анализировать динамику"
4. Смотрите график изменения энтропии

## Использование API

### Swagger UI (интерактивная документация)

Откройте http://localhost:8000/docs и тестируйте endpoints прямо в браузере!

### cURL примеры

```bash
# Анализ текста
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Мой дядя самых честных правил",
    "include_bootstrap": false
  }' | jq

# Сравнение текстов
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "Мой дядя самых честных правил",
    "text2": "Белеет парус одинокий",
    "name1": "Пушкин",
    "name2": "Лермонтов"
  }' | jq

# Загрузка файла
curl -X POST http://localhost:8000/api/v1/analyze/file \
  -F "file=@../entropy-analysis/text/pushkin.txt" \
  -F "include_bootstrap=true" | jq
```

## Использование CLI

```bash
# Простой анализ
uv run entropy-analysis analyze text.txt

# С детальным выводом
uv run entropy-analysis analyze text.txt --verbose

# С bootstrap CI
uv run entropy-analysis analyze text.txt --bootstrap

# Вывод в JSON
uv run entropy-analysis analyze text.txt --json result.json

# Пакетный анализ директории
uv run entropy-analysis batch ../entropy-analysis/text/ \
  --pattern "*.txt" \
  --csv results.csv

# Сравнение двух файлов
uv run entropy-analysis compare \
  ../entropy-analysis/text/pushkin.txt \
  ../entropy-analysis/text/lermontov_new.txt
```

## Использование Python API

```python
from entropy_analysis import TextAnalyzer

# 1. Простой анализ
analyzer = TextAnalyzer.create()
result = analyzer.analyze("Мой дядя самых честных правил")

print(f"Энтропия: {result.shannon_entropy:.4f} бит")

# 2. Пакетный анализ
texts = [
    ("pushkin", pushkin_text),
    ("lermontov", lermontov_text),
]
batch = analyzer.analyze_batch(texts)

print(f"Корреляция: {batch.correlation:.4f}")

# 3. Сравнение
comparison = analyzer.compare(text1, text2, "Автор1", "Автор2")
print(f"JS Divergence: {comparison.js_divergence:.4f}")

# 4. Визуализация
from entropy_analysis.visualization import create_letter_distribution_chart

fig = create_letter_distribution_chart(result)
fig.show()  # Откроется в браузере
# или
fig.write_html("chart.html")
```

## Миграция из Rust версии

### Анализ тех же файлов

```bash
# Был Rust:
cd entropy-analysis
cargo run -- analyze text/pushkin.txt

# Стал Python:
cd entropy-analysis-py
uv run entropy-analysis analyze ../entropy-analysis/text/pushkin.txt
```

### Результаты идентичны

Проверка на файле pushkin.txt:

**Rust версия:**
- N = 96,736
- H ≈ 4.254 bits

**Python версия:**
- N = 96,736
- H = 4.2541 bits ✅

## Проверка установки

Выполните:

```bash
# Тесты
make test

# Пример
make example

# CLI
uv run entropy-analysis version
```

Если всё прошло успешно - готово к работе! 🎉

## Troubleshooting

### Ошибка импорта

```bash
# Переустановить
uv sync --reinstall
```

### Порт занят

```bash
# Использовать другой порт
uv run entropy-analysis dashboard --port 8502
```

### Docker не запускается

```bash
# Проверить логи
docker logs entropy-dashboard

# Пересобрать
docker-compose build --no-cache
docker-compose up
```

## Дальнейшее обучение

1. Смотрите `example_usage.py` - 7 практических примеров
2. Читайте `README.md` - полная документация
3. Изучайте `SUMMARY.md` - что реализовано
4. Открывайте `/docs` в API - Swagger UI

**Приятного анализа! 📊**

