# Фильтрация данных на графиках корреляции

## Обзор

Теперь программа поддерживает фильтрацию точек на графиках корреляции по следующим параметрам:

- **`--min-entropy <значение>`** - минимальное значение энтропии H (в битах)
- **`--max-n-words <значение>`** - максимальное количество слов N

Эти параметры доступны для команд `batch` и `multi-poem`.

## Примеры использования

### Batch режим

#### Базовый анализ с графиком корреляции
```bash
cargo run --release -- batch test_poems/ \
    --correlation-plot correlation.svg
```

#### Фильтрация точек с низкой энтропией
```bash
cargo run --release -- batch test_poems/ \
    --correlation-plot correlation_filtered.svg \
    --min-entropy 2.5
```

#### Фильтрация точек с большим количеством слов
```bash
cargo run --release -- batch test_poems/ \
    --correlation-plot correlation_filtered.svg \
    --max-n-words 800
```

#### Комбинированная фильтрация
```bash
cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot correlation_filtered.svg \
    --min-entropy 2.5 \
    --max-n-words 8000
```

### Multi-poem режим (один автор)

```bash
cargo run --release -- multi-poem pushkin.txt \
    --correlation-plot pushkin_correlation.svg \
    --min-entropy 3.0 \
    --max-n-words 500
```

### Multi-poem режим (два автора)

```bash
cargo run --release -- multi-poem pushkin.txt \
    --second-file lermontov.txt \
    --correlation-plot dual_author_correlation.svg \
    --min-entropy 2.5 \
    --max-n-words 1000
```

## Что изменилось

### 1. Информативные легенды
Теперь легенды на графиках показывают количество отображаемых точек:
- Для одного автора: `Correlation: H vs N (r = 0.3234, R² = 0.1046), n = 450`
- Для двух авторов: `Пушкин (n=250)` и `Лермонтов (n=200)`

### 2. Фильтрация в консольном выводе
При использовании фильтров программа выводит информацию о примененных ограничениях:
```
Filters applied: min_entropy=Some(2.5), max_n_words=Some(800)
Samples: 180
Correlation coefficient (r): 0.456789
...
```

### 3. Автоматический пересчет корреляции
При применении фильтров корреляция пересчитывается только для отфильтрованных точек, что позволяет:
- Исключить выбросы из анализа
- Сосредоточиться на интересующем диапазоне значений
- Получить более точные результаты для подмножества данных

## Технические детали

### Структура FilterParams
Внутри программы используется структура `FilterParams`:
```rust
pub struct FilterParams {
    pub min_entropy: Option<f64>,
    pub max_n_words: Option<u64>,
}
```

### Функции фильтрации
- `calculate_correlation_with_filter()` - вычисление корреляции с фильтрацией
- `plot_correlation_scatter_with_filter()` - построение графика корреляции с фильтрацией
- `plot_dual_author_correlation_with_filter()` - построение двухавторского графика с фильтрацией

## Примечания

1. Фильтры применяются **только** к графикам и расчетам корреляции
2. CSV и JSON файлы содержат **все** данные без фильтрации
3. Если после фильтрации остается менее 2 точек, выводится предупреждение
4. Фильтры работают независимо - можно использовать только один или оба сразу
