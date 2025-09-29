#!/bin/bash
# Примеры использования фильтрации данных на графиках

echo "=== Примеры использования новых функций фильтрации ==="
echo ""

echo "1. Базовый анализ без фильтрации:"
echo "   cargo run --release -- batch test_poems/ --correlation --correlation-plot base.svg"
echo ""

echo "2. Фильтрация по минимальной энтропии (H >= 2.5):"
echo "   cargo run --release -- batch test_poems/ --correlation --correlation-plot filtered_min_h.svg --min-entropy 2.5"
echo ""

echo "3. Фильтрация по максимальному количеству слов (N <= 800):"
echo "   cargo run --release -- batch test_poems/ --correlation --correlation-plot filtered_max_n.svg --max-n-words 800"
echo ""

echo "4. Комбинированная фильтрация (H >= 2.5 И N <= 8000):"
echo "   cargo run --release -- batch test_poems/ --correlation --correlation-plot filtered_both.svg --min-entropy 2.5 --max-n-words 8000"
echo ""

echo "5. Multi-poem с фильтрацией:"
echo "   cargo run --release -- multi-poem pushkin.txt --correlation-plot pushkin_filtered.svg --min-entropy 3.0 --max-n-words 500"
echo ""

echo "6. Два автора с фильтрацией:"
echo "   cargo run --release -- multi-poem pushkin.txt --second-file lermontov.txt --correlation-plot dual_filtered.svg --min-entropy 2.5 --max-n-words 1000"
echo ""

echo "=== Что изменилось ==="
echo "✓ Легенды теперь показывают количество точек (n=X)"
echo "✓ Можно фильтровать точки по минимальной энтропии (--min-entropy)"
echo "✓ Можно фильтровать точки по максимальному количеству слов (--max-n-words)"
echo "✓ Фильтры работают для batch и multi-poem команд"
echo "✓ В консольном выводе показывается информация о примененных фильтрах"
echo ""

echo "=== Полезные советы ==="
echo "• Используйте --min-entropy для исключения точек с низкой энтропией"
echo "• Используйте --max-n-words для исключения длинных текстов"
echo "• Комбинируйте оба фильтра для более точного анализа"
echo "• CSV/JSON файлы всегда содержат все данные без фильтрации"
echo ""
