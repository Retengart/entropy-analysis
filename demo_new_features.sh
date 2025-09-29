#!/bin/bash
# Демонстрация новых возможностей программы анализа энтропии
# Версия 2.0

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Демонстрация новых возможностей программы анализа энтропии   ║"
echo "║                        Версия 2.0                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Проверка наличия тестовых данных
if [ ! -d "test_poems" ]; then
    echo "⚠️  Директория test_poems/ не найдена"
    echo "Создайте тестовые данные для демонстрации"
    exit 1
fi

echo "📁 Используется директория: test_poems/"
echo ""

# Создать директорию для результатов
DEMO_DIR="demo_results"
mkdir -p "$DEMO_DIR"
echo "📂 Результаты будут сохранены в: $DEMO_DIR/"
echo ""

# ============================================================================
# ЧАСТЬ 1: Базовый анализ без фильтров
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 ЧАСТЬ 1: Базовый анализ (без фильтров)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Команда:"
echo "  cargo run --release -- batch test_poems/ \\"
echo "      --correlation \\"
echo "      --correlation-plot $DEMO_DIR/01_baseline.svg"
echo ""

cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot "$DEMO_DIR/01_baseline.svg"

echo ""
echo "✅ График сохранен: $DEMO_DIR/01_baseline.svg"
echo "   Это базовая линия - все данные без фильтрации"
echo ""
sleep 2

# ============================================================================
# ЧАСТЬ 2: Фильтрация по минимальной энтропии
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 ЧАСТЬ 2: Фильтрация по минимальной энтропии"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Команда:"
echo "  cargo run --release -- batch test_poems/ \\"
echo "      --correlation \\"
echo "      --correlation-plot $DEMO_DIR/02_min_entropy.svg \\"
echo "      --min-entropy 2.5"
echo ""
echo "Исключаем тексты с энтропией H < 2.5"
echo ""

cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot "$DEMO_DIR/02_min_entropy.svg" \
    --min-entropy 2.5

echo ""
echo "✅ График сохранен: $DEMO_DIR/02_min_entropy.svg"
echo "   Исключены тексты с низкой энтропией"
echo ""
sleep 2

# ============================================================================
# ЧАСТЬ 3: Фильтрация по максимальному количеству слов
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📏 ЧАСТЬ 3: Фильтрация по максимальному количеству слов"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Команда:"
echo "  cargo run --release -- batch test_poems/ \\"
echo "      --correlation \\"
echo "      --correlation-plot $DEMO_DIR/03_max_n_words.svg \\"
echo "      --max-n-words 800"
echo ""
echo "Исключаем тексты с количеством слов N > 800"
echo ""

cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot "$DEMO_DIR/03_max_n_words.svg" \
    --max-n-words 800

echo ""
echo "✅ График сохранен: $DEMO_DIR/03_max_n_words.svg"
echo "   Исключены длинные тексты"
echo ""
sleep 2

# ============================================================================
# ЧАСТЬ 4: Комбинированная фильтрация
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 ЧАСТЬ 4: Комбинированная фильтрация (4 параметра)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Команда:"
echo "  cargo run --release -- batch test_poems/ \\"
echo "      --correlation \\"
echo "      --correlation-plot $DEMO_DIR/04_combined_filters.svg \\"
echo "      --min-entropy 2.5 \\"
echo "      --max-entropy 4.5 \\"
echo "      --min-n-words 50 \\"
echo "      --max-n-words 2000"
echo ""
echo "Диапазоны: 2.5 ≤ H ≤ 4.5 и 50 ≤ N ≤ 2000"
echo ""

cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot "$DEMO_DIR/04_combined_filters.svg" \
    --min-entropy 2.5 \
    --max-entropy 4.5 \
    --min-n-words 50 \
    --max-n-words 2000

echo ""
echo "✅ График сохранен: $DEMO_DIR/04_combined_filters.svg"
echo "   Показаны только тексты в заданных диапазонах"
echo ""
sleep 2

# ============================================================================
# ЧАСТЬ 5: Подписывание точек (топ-3 по энтропии)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏷️  ЧАСТЬ 5: Подписывание топ-3 текстов по энтропии"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Команда:"
echo "  cargo run --release -- batch test_poems/ \\"
echo "      --correlation \\"
echo "      --correlation-plot $DEMO_DIR/05_labeled_top3.svg \\"
echo "      --label-top-entropy 3"
echo ""
echo "Подписываем 3 текста с максимальной энтропией"
echo ""

cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot "$DEMO_DIR/05_labeled_top3.svg" \
    --label-top-entropy 3

echo ""
echo "✅ График сохранен: $DEMO_DIR/05_labeled_top3.svg"
echo "   Топ-3 текста выделены красным и подписаны"
echo ""
sleep 2

# ============================================================================
# ЧАСТЬ 6: Фильтрация + Подписывание
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎨 ЧАСТЬ 6: Комбинация - Фильтрация + Подписывание"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Команда:"
echo "  cargo run --release -- batch test_poems/ \\"
echo "      --correlation \\"
echo "      --correlation-plot $DEMO_DIR/06_filtered_labeled.svg \\"
echo "      --min-entropy 2.5 \\"
echo "      --max-n-words 1500 \\"
echo "      --label-top-entropy 5"
echo ""
echo "Фильтруем (H≥2.5, N≤1500) и подписываем топ-5"
echo ""

cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot "$DEMO_DIR/06_filtered_labeled.svg" \
    --min-entropy 2.5 \
    --max-n-words 1500 \
    --label-top-entropy 5

echo ""
echo "✅ График сохранен: $DEMO_DIR/06_filtered_labeled.svg"
echo "   Отфильтровано + топ-5 подписаны"
echo ""
sleep 2

# ============================================================================
# ЧАСТЬ 7: Полный анализ с экспортом
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💾 ЧАСТЬ 7: Полный анализ с экспортом в CSV и JSON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Команда:"
echo "  cargo run --release -- batch test_poems/ \\"
echo "      --correlation \\"
echo "      --correlation-plot $DEMO_DIR/07_full_analysis.svg \\"
echo "      --summary-csv $DEMO_DIR/results.csv \\"
echo "      --summary-json $DEMO_DIR/results.json \\"
echo "      --min-entropy 2.5 \\"
echo "      --max-entropy 4.5 \\"
echo "      --min-n-words 50 \\"
echo "      --max-n-words 3000 \\"
echo "      --label-top-entropy 5"
echo ""

cargo run --release -- batch test_poems/ \
    --correlation \
    --correlation-plot "$DEMO_DIR/07_full_analysis.svg" \
    --summary-csv "$DEMO_DIR/results.csv" \
    --summary-json "$DEMO_DIR/results.json" \
    --min-entropy 2.5 \
    --max-entropy 4.5 \
    --min-n-words 50 \
    --max-n-words 3000 \
    --label-top-entropy 5

echo ""
echo "✅ Файлы сохранены:"
echo "   - График: $DEMO_DIR/07_full_analysis.svg"
echo "   - CSV:    $DEMO_DIR/results.csv"
echo "   - JSON:   $DEMO_DIR/results.json"
echo ""
sleep 2

# ============================================================================
# ИТОГИ
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📂 Все результаты сохранены в директории: $DEMO_DIR/"
echo ""
echo "Созданные файлы:"
ls -lh "$DEMO_DIR/"
echo ""
echo "🎯 Что было продемонстрировано:"
echo ""
echo "   1. ✅ Базовый анализ без фильтров"
echo "   2. ✅ Фильтрация по минимальной энтропии (--min-entropy)"
echo "   3. ✅ Фильтрация по максимальному количеству слов (--max-n-words)"
echo "   4. ✅ Комбинированная фильтрация (все 4 параметра)"
echo "   5. ✅ Подписывание топ-N текстов (--label-top-entropy)"
echo "   6. ✅ Комбинация фильтрации и подписывания"
echo "   7. ✅ Полный анализ с экспортом в CSV/JSON"
echo ""
echo "📖 Для подробной информации см.:"
echo "   - NEW_FEATURES.md - описание новых возможностей"
echo "   - USAGE_EXAMPLES.md - примеры использования"
echo "   - IMPROVEMENT_IDEAS.md - идеи по улучшению"
echo "   - RELEASE_NOTES.md - полный список изменений"
echo ""
echo "🚀 Программа готова к использованию!"
echo ""
