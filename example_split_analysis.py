#!/usr/bin/env python3
"""
Пример использования функции split_and_analyze.

Демонстрирует анализ текста с разделителями и создание графика
зависимости энтропии от количества слов с линейной регрессией.
"""

from pathlib import Path

from entropy_analysis import TextAnalyzer
from entropy_analysis.visualization import create_correlation_scatter


def example_split_from_file():
    """Пример: Анализ текста из файла с разделителями."""
    print("=" * 70)
    print("ПРИМЕР: Анализ текста с разделителями из файла")
    print("=" * 70)

    # Создать анализатор
    analyzer = TextAnalyzer.create(alphabet="rus29")

    # Прочитать файл с текстами
    file_path = Path("example_split_text.txt")
    if not file_path.exists():
        print(f"❌ Файл {file_path} не найден")
        return

    text = file_path.read_text(encoding="utf-8")

    # Разделить и проанализировать
    print("\n📊 Анализируем сегменты текста...")
    batch = analyzer.split_and_analyze(text, delimiter="***")

    print(f"\n✅ Найдено и проанализировано сегментов: {len(batch.results)}")

    # Показать результаты каждого сегмента
    print("\n" + "─" * 70)
    print(f"{'#':<4} {'Сегмент':<20} {'N':>8} {'H (энтропия)':>15} {'x̄':>8}")
    print("─" * 70)

    for i, (name, result) in enumerate(batch.results, 1):
        h_str = f"{result.shannon_entropy:.4f}" if result.shannon_entropy else "—"
        mean_str = f"{result.mean_rank:.2f}" if result.mean_rank else "—"
        print(f"{i:<4} {name:<20} {result.n_words:>8,} {h_str:>15} {mean_str:>8}")

    print("─" * 70)

    # Статистика энтропии
    if batch.extended_stats:
        stats = batch.extended_stats
        print("\n📈 Статистика энтропии:")
        print(f"   Среднее:  {stats.mean:.4f} бит")
        print(f"   Медиана:  {stats.median:.4f} бит")
        print(f"   Стд.откл: {stats.std_dev:.4f}")
        print(f"   Диапазон: [{stats.min_value:.4f}, {stats.max_value:.4f}]")

    # Корреляция H vs N
    if batch.correlation is not None:
        print("\n📊 Корреляция: Энтропия (H) vs Количество слов (N)")
        print(f"   Коэффициент корреляции (r):  {batch.correlation:.4f}")
        print(f"   Коэффициент детерминации (R²): {batch.correlation_r_squared:.4f}")
        print(f"   p-value: {batch.correlation_p_value:.6f}")
        print("   Уравнение регрессии:")
        print(f"      H = {batch.correlation_slope:.6f} × N + {batch.correlation_intercept:.4f}")

        # Интерпретация
        print("\n💡 Интерпретация:")
        if abs(batch.correlation) > 0.7:
            strength = "сильная"
        elif abs(batch.correlation) > 0.4:
            strength = "умеренная"
        else:
            strength = "слабая"

        direction = "положительная" if batch.correlation > 0 else "отрицательная"
        print(f"   Обнаружена {strength} {direction} корреляция")

        if batch.correlation_p_value < 0.05:
            print("   ✓ Корреляция статистически значима (p < 0.05)")
        else:
            print("   ⚠ Корреляция статистически незначима (p ≥ 0.05)")

    # Создать график
    print("\n📊 Создание графика...")
    fig = create_correlation_scatter(
        batch,
        show_trendline=True,
        highlight_outliers=True,
        label_top_n=3,
    )

    # Обновить заголовок
    fig.update_layout(
        title=f"Энтропия vs Количество слов<br><sub>{len(batch.results)} сегментов из {file_path}</sub>"
    )

    # Сохранить график
    output_file = Path("output_split_analysis.html")
    fig.write_html(str(output_file))
    print(f"✅ График сохранён: {output_file}")

    # Открыть в браузере
    import webbrowser

    webbrowser.open(f"file://{output_file.absolute()}")
    print("🌐 График открыт в браузере")

    print("\n✅ Пример завершён успешно!\n")


def example_split_inline():
    """Пример: Анализ текста с разделителями напрямую из строки."""
    print("=" * 70)
    print("ПРИМЕР: Анализ текста с разделителями из строки")
    print("=" * 70)

    # Текст с разными сегментами
    text = """
    Короткий текст для анализа первый сегмент

    ***

    Более длинный текст для анализа энтропии во втором сегменте текста
    который содержит больше слов и должен иметь другую энтропию

    ***

    Третий сегмент текста среднего размера для сравнительного анализа
    с предыдущими сегментами по метрикам энтропии

    ***

    Четвёртый сегмент с минимальным количеством слов

    ***

    Пятый и последний сегмент текста который будет проанализирован
    вместе со всеми остальными сегментами для получения корреляции
    между количеством слов и значением энтропии распределения
    """

    # Создать анализатор
    analyzer = TextAnalyzer.create(alphabet="rus29")

    # Разделить и проанализировать
    batch = analyzer.split_and_analyze(text, delimiter="***")

    print(f"\n✅ Проанализировано сегментов: {len(batch.results)}")

    # Показать результаты
    for i, (name, result) in enumerate(batch.results, 1):
        print(f"\n{i}. {name}:")
        print(f"   N = {result.n_words:,} слов")
        if result.shannon_entropy:
            print(f"   H = {result.shannon_entropy:.4f} бит")

    # Корреляция
    if batch.correlation is not None:
        print(f"\n📈 Корреляция (r): {batch.correlation:.4f}")
        print(f"   R²: {batch.correlation_r_squared:.4f}")

    print("\n✅ Пример завершён!\n")


def example_custom_delimiter():
    """Пример: Использование пользовательского разделителя."""
    print("=" * 70)
    print("ПРИМЕР: Пользовательский разделитель")
    print("=" * 70)

    text = """
    Первый текст для анализа
    ---
    Второй текст для анализа который длиннее
    ---
    Третий текст
    """

    analyzer = TextAnalyzer.create()
    batch = analyzer.split_and_analyze(text, delimiter="---")

    print(f"\n✅ Сегментов с разделителем '---': {len(batch.results)}")

    for name, result in batch.results:
        print(f"\n{name}: N={result.n_words}, H={result.shannon_entropy:.4f}")

    print("\n✅ Пример завершён!\n")


def main():
    """Запуск всех примеров."""
    print("\n" + "🎓 " * 30)
    print("   ПРИМЕРЫ SPLIT-АНАЛИЗА С РАЗДЕЛИТЕЛЯМИ")
    print("🎓 " * 30 + "\n")

    try:
        # Основной пример с файлом
        example_split_from_file()

        # Дополнительные примеры
        example_split_inline()
        example_custom_delimiter()

        print("=" * 70)
        print("🎉 ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 70)
        print("\n📚 Использование из командной строки:")
        print("   uv run entropy-analysis split-analyze example_split_text.txt")
        print("   uv run entropy-analysis split-analyze example_split_text.txt --show")
        print("   uv run entropy-analysis split-analyze example_split_text.txt --delimiter='---'")
        print("\n")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
