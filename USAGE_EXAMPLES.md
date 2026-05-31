# Примеры использования - Полное руководство

## 📋 Оглавление
1. [Базовые команды](#базовые-команды)
2. [Фильтрация данных](#фильтрация-данных)
3. [Подписывание точек](#подписывание-точек)
4. [Комплексные сценарии](#комплексные-сценарии)

## Базовые команды

### Batch режим

#### Простой анализ директории
```bash
cargo run --release -- batch test_poems/ --correlation
```

#### С графиком корреляции
```bash
cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot correlation.svg
```

#### С сохранением результатов
```bash
cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot plot.svg \
    --summary-csv results.csv \
    --summary-json results.json
```

### Multi-poem режим

#### Один автор
```bash
cargo run --release -- multi-poem pushkin.txt \
    --delimiter "***" \
    --correlation-plot pushkin.svg
```

#### Два автора
```bash
cargo run --release -- multi-poem pushkin.txt \
    --second-file lermontov.txt \
    --correlation-plot comparison.svg
```

## Фильтрация данных

### Фильтры по энтропии

#### Минимальная энтропия
```bash
# Показать только тексты с H >= 2.5
cargo run --release -- batch test_poems/ \
    --correlation-plot filtered.svg \
    --min-entropy 2.5
```

#### Максимальная энтропия
```bash
# Показать только тексты с H <= 4.5
cargo run --release -- batch test_poems/ \
    --correlation-plot filtered.svg \
    --max-entropy 4.5
```

#### Диапазон энтропии
```bash
# Показать тексты с 2.5 <= H <= 4.5
cargo run --release -- batch test_poems/ \
    --correlation-plot filtered.svg \
    --min-entropy 2.5 \
    --max-entropy 4.5
```

### Фильтры по количеству слов

#### Минимальное количество слов
```bash
# Показать только тексты с N >= 50
cargo run --release -- batch test_poems/ \
    --correlation-plot filtered.svg \
    --min-n-words 50
```

#### Максимальное количество слов
```bash
# Показать только тексты с N <= 1000
cargo run --release -- batch test_poems/ \
    --correlation-plot filtered.svg \
    --max-n-words 1000
```

#### Диапазон количества слов
```bash
# Показать тексты с 50 <= N <= 1000
cargo run --release -- batch test_poems/ \
    --correlation-plot filtered.svg \
    --min-n-words 50 \
    --max-n-words 1000
```

### Комбинированные фильтры

#### Фокус на стихах средней длины
```bash
cargo run --release -- batch poetry/ \
    --correlation-plot medium_poems.svg \
    --min-n-words 30 \
    --max-n-words 300 \
    --min-entropy 2.5 \
    --max-entropy 4.2
```

#### Исключение выбросов
```bash
cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot no_outliers.svg \
    --min-entropy 2.0 \
    --max-entropy 4.5 \
    --max-n-words 5000
```

## Подписывание точек

### Топ-3 по энтропии
```bash
cargo run --release -- batch test_poems/ \
    --correlation-plot labeled.svg \
    --label-top-entropy 3
```

### Топ-5 с фильтрацией
```bash
# Среди отфильтрованных текстов показать топ-5
cargo run --release -- batch test_poems/ \
    --correlation-plot filtered_labeled.svg \
    --min-entropy 2.5 \
    --max-n-words 1000 \
    --label-top-entropy 5
```

### Большое количество меток
```bash
# Для больших датасетов
cargo run --release -- batch large_corpus/ \
    --correlation-plot top10.svg \
    --label-top-entropy 10
```

## Комплексные сценарии

### Полный анализ с выводом всех данных

```bash
cargo run --release -- batch poems/ \
    --recurse \
    --ext txt \
    --correlation \
    --correlation-plot analysis.svg \
    --summary-csv summary.csv \
    --summary-json summary.json \
    --min-entropy 2.5 \
    --max-entropy 4.5 \
    --min-n-words 50 \
    --max-n-words 2000 \
    --label-top-entropy 5
```

**Что происходит:**
1. Рекурсивно обходит директорию `poems/`
2. Находит все `.txt` файлы
3. Вычисляет корреляцию
4. Строит график с фильтрацией: 2.5 ≤ H ≤ 4.5, 50 ≤ N ≤ 2000
5. Подписывает топ-5 текстов по энтропии
6. Сохраняет результаты в CSV и JSON (без фильтрации)

### Сравнение двух авторов с детальным анализом

```bash
cargo run --release -- multi-poem pushkin_complete.txt \
    --second-file lermontov_complete.txt \
    --delimiter "***" \
    --correlation \
    --correlation-plot pushkin_vs_lermontov.svg \
    --normal-dist-plot distributions.svg \
    --analysis-table analysis.csv \
    --summary-csv summary.csv \
    --min-entropy 2.5 \
    --max-n-words 1500 \
    --label-top-entropy 3
```

**Результат:**
- Двухцветный график корреляции (Пушкин синий, Лермонтов красный)
- По 3 подписанных текста от каждого автора
- График нормального распределения
- Подробная таблица анализа
- Сводная таблица в CSV

### Анализ корреляции между системами перевода

```bash
cargo run --release -- multi-poem translations.txt \
    --delimiter "***" \
    --correlation-lang translation_correlation.svg
```

**Результат:**
- График корреляции между разными системами перевода (Яндекс, Google, DeepL и др.)
- Показывает, насколько согласованы энтропийные характеристики текстов, переведенных разными системами

### Комплексный анализ с графиками корреляции

```bash
cargo run --release -- multi-poem poems_with_translations.txt \
    --delimiter "***" \
    --correlation \
    --correlation-plot entropy_correlation.svg \
    --correlation-lang translation_correlation.svg \
    --normal-dist-plot entropy_distribution.svg \
    --analysis-table detailed_analysis.csv \
    --summary-csv summary.csv \
    --min-entropy 2.0 \
    --max-entropy 5.0 \
    --label-top-entropy 5
```

**Результат:**
- Стандартный график корреляции H vs N
- График корреляции между системами перевода
- График нормального распределения энтропии
- Подробная таблица анализа всех текстов
- Фильтрация текстов с энтропией от 2.0 до 5.0 бит
- Подпись топ-5 текстов по энтропии

### Анализ конкретного периода творчества

```bash
# Только ранние стихи (обычно короче)
cargo run --release -- batch pushkin_early/ \
    --correlation-plot early_period.svg \
    --max-n-words 200 \
    --label-top-entropy 5

# Только поздние стихи (могут быть длиннее)
cargo run --release -- batch pushkin_late/ \
    --correlation-plot late_period.svg \
    --min-n-words 100 \
    --label-top-entropy 5
```

### Поиск аномалий

```bash
# Найти тексты с экстремально высокой энтропией
cargo run --release -- batch corpus/ \
    --correlation-plot anomalies.svg \
    --min-entropy 4.2 \
    --label-top-entropy 10
```

### Фокус на качественных текстах

```bash
# Достаточно длинные, с нормальной энтропией
cargo run --release -- batch poems/ \
    --correlation-plot quality.svg \
    --min-n-words 100 \
    --min-entropy 2.5 \
    --max-entropy 4.2 \
    --label-top-entropy 5
```

## Workflow для научного исследования

### Шаг 1: Первичный анализ
```bash
# Получить общую картину без фильтров
cargo run --release -- batch corpus/ \
    --correlation \
    --correlation-plot full_data.svg \
    --summary-csv full_results.csv
```

### Шаг 2: Изучение распределения
```bash
# Посмотреть на данные, определить выбросы
# (изучить full_data.svg и full_results.csv)
```

### Шаг 3: Фильтрация выбросов
```bash
# На основе шага 2, отфильтровать аномалии
cargo run --release -- batch corpus/ \
    --correlation \
    --correlation-plot filtered_data.svg \
    --summary-csv filtered_results.csv \
    --min-entropy 2.5 \
    --max-entropy 4.3 \
    --max-n-words 3000 \
    --label-top-entropy 10
```

### Шаг 4: Детальный анализ
```bash
# Изучить топ-10 текстов с максимальной энтропией
# (посмотреть на подписи на filtered_data.svg)
# Проверить эти файлы вручную
```

### Шаг 5: Финальный отчет
```bash
# Создать финальные визуализации для публикации
cargo run --release -- batch corpus/ \
    --correlation-plot final_figure.svg \
    --summary-json final_results.json \
    --min-entropy 2.5 \
    --max-entropy 4.2 \
    --min-n-words 50 \
    --max-n-words 2000 \
    --label-top-entropy 5
```

## Работа с разными алфавитами

### Стандартный русский (28 букв)
```bash
cargo run --release -- batch texts/ \
    --alphabet rus28 \
    --correlation-plot rus28.svg
```

### Полный русский (33 буквы)
```bash
cargo run --release -- batch texts/ \
    --alphabet rus33 \
    --correlation-plot rus33.svg
```

### С сохранением ё и й
```bash
cargo run --release -- batch texts/ \
    --alphabet rus28 \
    --keep-yo \
    --keep-j \
    --correlation-plot with_yo_j.svg
```

## Полезные комбинации параметров

### Для анализа стихов
```bash
--min-n-words 20 --max-n-words 500 --min-entropy 2.5 --label-top-entropy 5
```

### Для анализа прозы
```bash
--min-n-words 500 --max-n-words 10000 --min-entropy 3.0 --label-top-entropy 3
```

### Для поиска коротких заметок/фрагментов
```bash
--max-n-words 100 --label-top-entropy 10
```

### Для исключения технических текстов
```bash
--min-entropy 2.0 --min-n-words 30
```

## Интерпретация результатов

### Высокая энтропия (H > 4.0)
- Разнообразная лексика
- Возможно, нетипичный для автора текст
- Может быть ошибкой/шумом в данных

### Низкая энтропия (H < 2.5)
- Однообразная лексика
- Короткий текст
- Технический/неполный текст

### Сильная корреляция (r > 0.5)
- Четкая зависимость H от N
- Стабильный стиль автора

### Слабая корреляция (r < 0.3)
- Разнородный корпус
- Смешение жанров/периодов
- Возможно, несколько авторов
