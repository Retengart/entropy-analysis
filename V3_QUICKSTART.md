# Быстрый старт - Версия 3.0

## 🚀 Новые возможности

Версия 3.0 добавляет три профессиональных функции:
1. **Расширенная статистика** - полный набор статистических метрик
2. **Определение выбросов** - три метода (IQR, Z-score, Modified Z-score)
3. **Нормализация энтропии** - корректное сравнение текстов разной длины

---

## ⚡ Быстрые команды

### Показать расширенную статистику
```bash
cargo run --release -- batch poems/ \
    --show-extended-stats
```

### Найти выбросы
```bash
cargo run --release -- batch poems/ \
    --auto-filter-outliers iqr
```

### Отметить выбросы на графике
```bash
cargo run --release -- batch poems/ \
    --mark-outliers iqr \
    --correlation-plot outliers.svg
```

### Нормализованная энтропия
```bash
cargo run --release -- batch poems/ \
    --normalized-entropy \
    --correlation-plot normalized.svg
```

### Полный анализ
```bash
cargo run --release -- batch poems/ \
    --show-extended-stats \
    --auto-filter-outliers iqr \
    --normalized-entropy \
    --mark-outliers iqr \
    --label-top-entropy 5 \
    --correlation-plot full_analysis.svg \
    --summary-json results.json
```

---

## 📊 Параметры расширенной статистики

| Параметр | Описание |
|----------|----------|
| `--show-extended-stats` | Показать медиану, квартили, IQR, CV, skewness, kurtosis |

**Пример вывода:**
```
=== Extended Statistics (Entropy H) ===
Median:      3.856234
Q1 (25%):    3.654321
Q3 (75%):    4.023456
IQR:         0.369135
Std Dev:     0.234567
CV (%):      6.12
Skewness:    -0.145678
Kurtosis:    0.234567
```

---

## 🔍 Параметры определения выбросов

### Автоматическая фильтрация

| Параметр | Значения | Описание |
|----------|----------|----------|
| `--auto-filter-outliers` | `iqr`, `z-score`, `modified-z-score` | Автоматически исключить выбросы |
| `--outlier-threshold` | число | Порог определения (по умолчанию: 1.5 для IQR, 3.0 для Z-score, 3.5 для Modified) |

### Визуальная отметка

| Параметр | Значения | Описание |
|----------|----------|----------|
| `--mark-outliers` | `iqr`, `z-score`, `modified-z-score` | Отметить выбросы на графике (пурпурным) |

**Примеры:**

```bash
# IQR метод (рекомендуемый)
--auto-filter-outliers iqr --outlier-threshold 1.5

# Z-score метод (для нормальных распределений)
--auto-filter-outliers z-score --outlier-threshold 3.0

# Modified Z-score (для асимметричных данных)
--auto-filter-outliers modified-z-score --outlier-threshold 3.5
```

---

## ⚖️ Параметры нормализации

| Параметр | Формула | Описание |
|----------|---------|----------|
| `--normalized-entropy` | H / log₂(N) | Нормализованная энтропия (0 до 1) |
| `--entropy-rate` | H / N | Энтропия на слово |

**Когда использовать:**
- `--normalized-entropy` - для сравнения текстов разной длины
- `--entropy-rate` - для теоретического анализа

---

## 🎨 Визуализация

### Цветовое кодирование на графиках

- **Синий** (●) - обычные точки
- **Пурпурный** (●) - выбросы (с `--mark-outliers`)
- **Красный** (●) - топ-N по энтропии (с `--label-top-entropy`)

### Размеры точек

- Обычные: 4px
- Выбросы: 5px
- Подписанные: 6px

---

## 💡 Типичные сценарии

### Сценарий 1: Быстрая проверка данных
```bash
cargo run --release -- batch new_data/ \
    --auto-filter-outliers iqr \
    --show-extended-stats
```

### Сценарий 2: Научная публикация
```bash
cargo run --release -- batch corpus/ \
    --show-extended-stats \
    --normalized-entropy \
    --mark-outliers modified-z-score \
    --label-top-entropy 3 \
    --correlation-plot figure1.svg \
    --summary-json table1.json
```

### Сценарий 3: Поиск аномалий
```bash
cargo run --release -- batch poems/ \
    --auto-filter-outliers iqr \
    --outlier-threshold 1.5 \
    --mark-outliers iqr
```

### Сценарий 4: Сравнение авторов
```bash
cargo run --release -- multi-poem pushkin.txt \
    --second-file lermontov.txt \
    --normalized-entropy \
    --show-extended-stats \
    --mark-outliers z-score \
    --correlation-plot comparison.svg
```

---

## 📈 Интерпретация результатов

### Расширенная статистика

**CV (Coefficient of Variation):**
- < 10% → низкая вариативность (однородный стиль)
- 10-20% → умеренная вариативность
- > 20% → высокая вариативность (смешанный корпус)

**Skewness (асимметрия):**
- > 0 → больше текстов с низкой энтропией
- < 0 → больше текстов с высокой энтропией
- |Skewness| < 0.5 → симметричное распределение

**Kurtosis (эксцесс):**
- > 0 → больше выбросов, чем в нормальном распределении
- < 0 → меньше выбросов
- |Kurtosis| < 1 → близко к нормальному

### Определение выбросов

**Количество выбросов:**
- < 5% → здоровый корпус
- 5-10% → требует внимания
- > 10% → проблемы с данными или смешение источников

**Типичные причины выбросов:**
- Технические тексты
- Ошибки обработки
- Фрагменты или незавершенные тексты
- Тексты на другом языке

### Нормализованная энтропия

**H_normalized:**
- > 0.8 → очень высокое разнообразие
- 0.6-0.8 → нормальное разнообразие для художественных текстов
- < 0.6 → низкое разнообразие (возможно, технические тексты)

---

## 🔧 Комбинации параметров

### Минимальный набор
```bash
--show-extended-stats
```

### Стандартный анализ
```bash
--show-extended-stats \
--mark-outliers iqr \
--correlation-plot result.svg
```

### Полный профессиональный анализ
```bash
--show-extended-stats \
--auto-filter-outliers iqr \
--normalized-entropy \
--entropy-rate \
--mark-outliers iqr \
--label-top-entropy 5 \
--correlation-plot analysis.svg \
--summary-json data.json \
--summary-csv data.csv
```

---

## ⚠️ Важные замечания

1. **Расширенная статистика** вычисляется для всего корпуса
2. **Автоматическая фильтрация** (`--auto-filter-outliers`) исключает выбросы из анализа
3. **Визуальная отметка** (`--mark-outliers`) только показывает, не исключает
4. **Нормализация** применяется к каждому тексту отдельно
5. **CSV/JSON** файлы всегда содержат все данные без фильтрации

---

## 🆘 Troubleshooting

### "Insufficient data for extended stats"
- Требуется минимум 2 текста
- Увеличьте размер корпуса

### "No outliers detected"
- Попробуйте уменьшить `--outlier-threshold`
- Данные могут быть очень однородными
- Это нормально для качественного корпуса

### "Too many outliers"
- Проверьте корпус на техническиетексты
- Попробуйте `modified-z-score` вместо `iqr`
- Увеличьте порог определения

---

## 📚 Полная документация

- **ADVANCED_FEATURES.md** - подробное руководство по новым функциям
- **USAGE_EXAMPLES.md** - больше примеров использования
- **IMPROVEMENT_IDEAS.md** - будущие улучшения

---

**Версия:** 3.0  
**Готово к использованию!** ✅
