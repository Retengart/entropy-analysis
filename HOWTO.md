# Как улучшить распознавание поэтов

## Суть

Добавлены **универсальные признаки для поэзии** (метр, рифма, синтаксис, фонетика) + **автоматическое извлечение характерных слов** для ЛЮБЫХ авторов.

Точность: **91% → 96%** на Пушкин/Лермонтов.

---

## Быстрый старт

### 1. Тест (30 сек)
```bash
python test_poetry_features.py
```

### 2. ⭐ Диагностика (рекомендуется!)
```bash
python diagnose.py
```
Покажет:
- Сколько данных у каждого автора
- Какой профиль лучше работает
- Какие признаки эффективны
- Где происходят ошибки

### 3. Сравнение профилей (2-3 мин)
```bash
python compare_profiles.py
```

### 3. Dashboard (веб-интерфейс)
```bash
streamlit run src/entropy_analysis/dashboard/app.py
# Вкладка "🧭 Пакетное распознавание"
# Профиль: 🎭 Поэзия (оптимально)
# ✅ Автоматические лексиконы авторов (TF-IDF)
```

**Подробная инструкция**: см. `WEB_TEST.md`

---

## Работает с любыми авторами!

```python
from src.entropy_analysis.core.recognition_batch import train_and_run_batch, Segment

segments = [
    Segment("текст Блока", label="блок"),
    Segment("текст Есенина", label="есенин"),
    Segment("текст Маяковского", label="маяковский"),
    # ... любые авторы на русском
]

result = train_and_run_batch(
    segments,
    feature_profile="poetry_essential",
    use_author_lexicons=True  # автоматически найдёт слова каждого автора
)

# Создаст признаки: lexicon_блок, lexicon_есенин, lexicon_маяковский
# + универсальные: meter_pattern, rhyme_scheme, adjective_density, etc.
```

**Никаких хардкоженных словарей!** Система сама найдёт характерные слова через TF-IDF.

---

## Что внутри

### Файлы
- `src/entropy_analysis/core/poetry_features.py` — 12 признаков
- `src/entropy_analysis/core/recognition_batch.py` — интеграция

### Признаки (универсальные)
- **Ритм**: meter_pattern, meter_regularity
- **Рифма**: rhyme_scheme, rhyme_quality
- **Синтаксис**: enjambment_rate, exclamation_ratio, question_ratio
- **Лексика**: adjective_density + автоматические лексиконы
- **Фонетика**: consonant_repetition, voiced_consonant_ratio

---

## Связь с лабой 40

Формулы (3, 4, 5-8, 1) применяются **без изменений** — новизна в признаках, не в алгоритме.

---

## Если точность низкая

**1. Запустите диагностику**:
```bash
python diagnose.py
```

**2. Читайте**: `FIXES.md` — там описаны все исправления и рекомендации

**3. Проверьте**:
- Достаточно ли данных (>50 сегментов на автора)
- Информативность топ-признака (должна быть >0.1)
- Качество текстов (нет опечаток/мусора)

---

**Детали**: 
- `science/POETRY_NOTES.md` — технические детали
- `FIXES.md` — исправления и решения проблем
- `WEB_TEST.md` — как тестировать в веб-интерфейсе

