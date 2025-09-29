# Быстрая справка по фильтрации

## Синтаксис

### Batch режим
```bash
cargo run --release -- batch <ДИРЕКТОРИЯ> \
    [--correlation] \
    [--correlation-plot <ФАЙЛ>] \
    [--min-entropy <ЧИСЛО>] \
    [--max-n-words <ЧИСЛО>]
```

### Multi-poem режим (один автор)
```bash
cargo run --release -- multi-poem <ФАЙЛ> \
    [--correlation] \
    [--correlation-plot <ФАЙЛ>] \
    [--min-entropy <ЧИСЛО>] \
    [--max-n-words <ЧИСЛО>]
```

### Multi-poem режим (два автора)
```bash
cargo run --release -- multi-poem <ФАЙЛ1> \
    --second-file <ФАЙЛ2> \
    [--correlation-plot <ФАЙЛ>] \
    [--min-entropy <ЧИСЛО>] \
    [--max-n-words <ЧИСЛО>]
```

## Параметры фильтрации

| Параметр | Тип | Описание | Пример |
|----------|-----|----------|--------|
| `--min-entropy` | f64 | Минимальная энтропия H (бит) | `--min-entropy 2.5` |
| `--max-n-words` | u64 | Максимальное количество слов N | `--max-n-words 800` |

## Типичные случаи использования

### 1. Исключить тексты с низкой энтропией
```bash
--min-entropy 2.5
```
**Когда использовать:** Если на графике много точек с H < 2.5, которые искажают корреляцию.

### 2. Исключить очень длинные тексты
```bash
--max-n-words 800
```
**Когда использовать:** Если несколько длинных текстов (N > 800) растягивают ось X.

### 3. Сфокусироваться на "нормальных" текстах
```bash
--min-entropy 2.5 --max-n-words 8000
```
**Когда использовать:** Для анализа текстов среднего размера с адекватной энтропией.

## Что показывают легенды

### Одноавторский график
```
Correlation: H vs N (r = 0.3234, R² = 0.1046), n = 450
                                                  ^^^^^^
                                         Количество точек
```

### Двухавторский график
```
Пушкин (n=250)    ← 250 текстов Пушкина отображено
Лермонтов (n=200) ← 200 текстов Лермонтова отображено
```

## Консольный вывод

### Без фильтрации
```
Correlation Analysis:
Samples: 500
Correlation coefficient (r): 0.323400
...
```

### С фильтрацией
```
Correlation Analysis:
Filters applied: min_entropy=Some(2.5), max_n_words=Some(800)
Samples: 180
Correlation coefficient (r): 0.456789
...
```

## Важные замечания

✅ **Применяется:** К графикам и расчетам корреляции
❌ **НЕ применяется:** К CSV/JSON выходным файлам
⚠️ **Предупреждение:** Если после фильтрации остается < 2 точек, будет выведено предупреждение

## Пример команды из вашего случая

Для графика с H > 2.5 и N < 8000:
```bash
cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot filtered_plot.svg \
    --min-entropy 2.5 \
    --max-n-words 8000
```

Или для multi-poem:
```bash
cargo run --release -- multi-poem pushkin.txt \
    --correlation-plot pushkin_filtered.svg \
    --min-entropy 2.5 \
    --max-n-words 8000
```
