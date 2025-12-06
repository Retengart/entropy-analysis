#!/usr/bin/env python3
"""
Примеры использования библиотеки entropy-analysis.

Демонстрирует основные возможности:
- Анализ одного текста
- Пакетный анализ
- Сравнение текстов
- Визуализация
"""

from entropy_analysis import TextAnalyzer
from entropy_analysis.visualization import (
    create_letter_distribution_chart,
    create_zipf_plot,
)


def example_1_simple_analysis():
    """Пример 1: Простой анализ одного текста."""
    print("=" * 60)
    print("ПРИМЕР 1: Анализ одного текста")
    print("=" * 60)

    # Создать анализатор
    analyzer = TextAnalyzer.create(alphabet="rus29")

    # Текст для анализа
    text = """
    Мой дядя самых честных правил,
    Когда не в шутку занемог,
    Он уважать себя заставил
    И лучше выдумать не мог.
    """

    # Анализ
    result = analyzer.analyze(text, source_name="Евгений Онегин (отрывок)")

    # Вывод результатов
    print("\n📊 Результаты:")
    print(f"   Слов: {result.n_words}")
    print(f"   Уникальных букв: {result.n_unique_letters}")
    print(f"   Энтропия H: {result.shannon_entropy:.4f} бит")
    print(f"   Средний ранг x̄: {result.mean_rank:.2f}")
    print(f"   Стд. откл. σ: {result.std_rank:.2f}")

    if result.simpson_index:
        print(f"   Simpson Index: {result.simpson_index:.4f}")
    if result.zipf_alpha:
        print(f"   Zipf α: {result.zipf_alpha:.3f}")

    print("\n✅ Пример 1 завершён\n")


def example_2_bootstrap_confidence():
    """Пример 2: Анализ с bootstrap доверительным интервалом."""
    print("=" * 60)
    print("ПРИМЕР 2: Bootstrap доверительный интервал")
    print("=" * 60)

    analyzer = TextAnalyzer.create()
    text = "Мой дядя самых честных правил " * 10  # Повторим для большей выборки

    # Анализ с bootstrap
    result = analyzer.analyze(
        text,
        include_bootstrap=True,
        bootstrap_iterations=1000,
    )

    print("\n📊 Энтропия с доверительным интервалом:")
    print(f"   H = {result.shannon_entropy:.4f} бит")

    if result.bootstrap:
        b = result.bootstrap
        print(f"   95% CI: [{b.ci_lower:.4f}, {b.ci_upper:.4f}]")
        print(f"   Стд. ошибка: ±{b.std_error:.4f}")
        print(f"   Bootstrap iterations: {b.n_bootstrap}")

    print("\n✅ Пример 2 завершён\n")


def example_3_compare_texts():
    """Пример 3: Сравнение двух текстов."""
    print("=" * 60)
    print("ПРИМЕР 3: Сравнение текстов")
    print("=" * 60)

    analyzer = TextAnalyzer.create()

    # Два разных текста
    text_pushkin = """
    Мой дядя самых честных правил,
    Когда не в шутку занемог,
    Он уважать себя заставил
    И лучше выдумать не мог.
    """

    text_lermontov = """
    Белеет парус одинокий
    В тумане моря голубом!..
    Что ищет он в стране далекой?
    Что кинул он в краю родном?..
    """

    # Сравнение
    comparison = analyzer.compare(text_pushkin, text_lermontov, "Пушкин", "Лермонтов")

    print("\n🔄 Результаты сравнения:")
    print(f"   KL(Пушкин || Лермонтов): {comparison.kl_divergence_p_q:.4f}")
    print(f"   KL(Лермонтов || Пушкин): {comparison.kl_divergence_q_p:.4f}")
    print(f"   JS Divergence: {comparison.js_divergence:.4f}")
    print(f"   Cosine Similarity: {comparison.cosine_similarity:.4f}")

    print("\n💡 Интерпретация:")
    if comparison.js_divergence < 0.1:
        print("   Тексты очень похожи")
    elif comparison.js_divergence < 0.3:
        print("   Тексты умеренно похожи")
    else:
        print("   Тексты значительно различаются")

    print("\n✅ Пример 3 завершён\n")


def example_4_batch_analysis():
    """Пример 4: Пакетный анализ."""
    print("=" * 60)
    print("ПРИМЕР 4: Пакетный анализ нескольких текстов")
    print("=" * 60)

    analyzer = TextAnalyzer.create()

    # Несколько текстов
    texts = [
        ("Текст 1 (короткий)", "Мой дядя самых честных правил"),
        ("Текст 2 (средний)", "Мой дядя самых честных правил когда не в шутку занемог"),
        (
            "Текст 3 (длинный)",
            "Мой дядя самых честных правил когда не в шутку занемог "
            "он уважать себя заставил и лучше выдумать не мог",
        ),
    ]

    # Пакетный анализ
    batch = analyzer.analyze_batch(texts)

    print(f"\n📁 Проанализировано текстов: {len(batch.results)}")

    # Результаты
    for name, result in batch.results:
        print(f"\n   {name}:")
        print(f"      N = {result.n_words}, H = {result.shannon_entropy:.4f}")

    # Корреляция
    if batch.correlation is not None:
        print("\n📈 Корреляция H vs N:")
        print(f"   r = {batch.correlation:.4f}")
        print(f"   R² = {batch.correlation_r_squared:.4f}")
        print(f"   p-value = {batch.correlation_p_value:.4f}")

    # Статистика
    if batch.extended_stats:
        s = batch.extended_stats
        print("\n📊 Статистика энтропии:")
        print(f"   Среднее: {s.mean:.4f}")
        print(f"   Медиана: {s.median:.4f}")
        print(f"   Стд. откл.: {s.std_dev:.4f}")

    print("\n✅ Пример 4 завершён\n")


def example_5_visualizations():
    """Пример 5: Создание графиков."""
    print("=" * 60)
    print("ПРИМЕР 5: Интерактивные визуализации")
    print("=" * 60)

    analyzer = TextAnalyzer.create()
    text = """
    Мой дядя самых честных правил,
    Когда не в шутку занемог,
    Он уважать себя заставил
    И лучше выдумать не мог.
    Его пример другим наука;
    Но, боже мой, какая скука
    С больным сидеть и день и ночь,
    Не отходя ни шагу прочь!
    """

    result = analyzer.analyze(text)

    print("\n📊 Создание графиков...")

    # График 1: Распределение букв
    create_letter_distribution_chart(result, sort_by="frequency", show_cumulative=True)
    print("   ✓ График распределения букв")

    # График 2: Закон Ципфа
    create_zipf_plot(result)
    print("   ✓ График закона Ципфа")

    # Сохранение (опционально)
    # fig1.write_html("letter_distribution.html")
    # fig2.write_image("zipf.png")

    print("\n💡 Чтобы посмотреть графики:")
    print("   1. Раскомментируйте fig.write_html() или fig.show()")
    print("   2. Или запустите dashboard: uv run entropy-analysis dashboard")

    print("\n✅ Пример 5 завершён\n")


def example_6_rolling_entropy():
    """Пример 6: Скользящая энтропия."""
    print("=" * 60)
    print("ПРИМЕР 6: Анализ динамики (скользящая энтропия)")
    print("=" * 60)

    analyzer = TextAnalyzer.create()

    # Длинный текст (повторим для примера)
    text = (
        """
    Мой дядя самых честных правил,
    Когда не в шутку занемог,
    Он уважать себя заставил
    И лучше выдумать не мог.
    """
        * 10
    )

    # Rolling entropy
    result = analyzer.rolling_entropy(text, window_size=20, step_size=5)

    print("\n📈 Результаты скользящей энтропии:")
    print(f"   Окно: {result.window_size} слов")
    print(f"   Позиций: {len(result.positions)}")
    print(f"   Среднее H: {result.mean_entropy:.4f}")
    print(f"   Стд. откл.: {result.std_entropy:.4f}")
    print(f"   Диапазон: [{result.min_entropy:.4f}, {result.max_entropy:.4f}]")

    print("\n✅ Пример 6 завершён\n")


def example_7_save_to_dataframe():
    """Пример 7: Сохранение результатов в DataFrame."""
    print("=" * 60)
    print("ПРИМЕР 7: Экспорт в Polars DataFrame")
    print("=" * 60)

    analyzer = TextAnalyzer.create()

    texts = [
        ("pushkin", "Мой дядя самых честных правил когда не в шутку занемог"),
        ("lermontov", "Белеет парус одинокий в тумане моря голубом"),
        ("blok", "Ночь улица фонарь аптека"),
    ]

    batch = analyzer.analyze_batch(texts)

    # Конвертация в DataFrame
    df = analyzer.batch_to_dataframe(batch)

    print("\n📋 DataFrame (preview):")
    print(df)

    # Сохранение
    # df.write_csv("results.csv")
    # df.write_parquet("results.parquet")

    print("\n💡 Можно сохранить в:")
    print("   - CSV: df.write_csv('results.csv')")
    print("   - Parquet: df.write_parquet('results.parquet')")
    print("   - JSON: df.write_json('results.json')")

    print("\n✅ Пример 7 завершён\n")


def main():
    """Запуск всех примеров."""
    print("\n" + "🎓 " * 20)
    print("   ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ENTROPY-ANALYSIS")
    print("🎓 " * 20 + "\n")

    try:
        example_1_simple_analysis()
        example_2_bootstrap_confidence()
        example_3_compare_texts()
        example_4_batch_analysis()
        example_5_visualizations()
        example_6_rolling_entropy()
        example_7_save_to_dataframe()

        print("=" * 60)
        print("🎉 ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 60)
        print("\n📚 Следующие шаги:")
        print("   1. Запустите веб-дашборд: uv run entropy-analysis dashboard")
        print("   2. Попробуйте API: uv run entropy-analysis serve")
        print("   3. Или используйте CLI: uv run entropy-analysis analyze file.txt")
        print("\n")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
