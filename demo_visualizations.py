#!/usr/bin/env python3
"""
Демонстрация всех типов визуализации.

Создаёт все 9 типов графиков и сохраняет в HTML для просмотра.
"""

from pathlib import Path

from entropy_analysis import TextAnalyzer
from entropy_analysis.visualization import (
    create_comparison_heatmap,
    create_correlation_scatter,
    create_dual_author_comparison,
    create_entropy_histogram,
    create_letter_distribution_chart,
    create_radar_chart,
    create_rolling_entropy_chart,
    create_zipf_plot,
)


def main():
    """Создать все графики и сохранить в HTML."""
    print("🎨 Демонстрация всех визуализаций")
    print("=" * 60)

    # Создать output директорию
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Создать анализатор
    analyzer = TextAnalyzer.create()

    # ==== График 1: Распределение букв ====
    print("\n1️⃣  Создание графика распределения букв...")
    text1 = """
    Мой дядя самых честных правил,
    Когда не в шутку занемог,
    Он уважать себя заставил
    И лучше выдумать не мог.
    Его пример другим наука;
    Но, боже мой, какая скука
    С больным сидеть и день и ночь,
    Не отходя ни шагу прочь!
    """
    result1 = analyzer.analyze(text1, source_name="Евгений Онегин")

    fig1 = create_letter_distribution_chart(result1, sort_by="alphabet", show_cumulative=True)
    fig1.write_html(output_dir / "1_letter_distribution.html")
    print("   ✓ Сохранено: demo_output/1_letter_distribution.html")

    # ==== График 2: По частоте ====
    print("\n2️⃣  Создание графика (сортировка по частоте)...")
    fig2 = create_letter_distribution_chart(result1, sort_by="frequency")
    fig2.write_html(output_dir / "2_sorted_frequency.html")
    print("   ✓ Сохранено: demo_output/2_sorted_frequency.html")

    # ==== График 3: Zipf's law ====
    print("\n3️⃣  Создание графика закона Ципфа...")
    fig3 = create_zipf_plot(result1)
    fig3.write_html(output_dir / "3_zipf_law.html")
    print("   ✓ Сохранено: demo_output/3_zipf_law.html")

    # ==== График 4: Пакетный анализ для корреляции ====
    print("\n4️⃣  Создание графика корреляции (batch)...")
    texts = [
        ("Короткий 1", "Мой дядя самых честных правил"),
        ("Короткий 2", "Белеет парус одинокий"),
        ("Средний 1", "Мой дядя самых честных правил когда не в шутку занемог"),
        ("Средний 2", "Белеет парус одинокий в тумане моря голубом"),
        ("Длинный", text1),
        ("Очень длинный", text1 * 3),
    ]
    batch = analyzer.analyze_batch(texts)

    fig4 = create_correlation_scatter(batch, show_trendline=True, label_top_n=3)
    fig4.write_html(output_dir / "4_correlation_scatter.html")
    print("   ✓ Сохранено: demo_output/4_correlation_scatter.html")

    # ==== График 5: Гистограмма энтропии ====
    print("\n5️⃣  Создание гистограммы энтропии...")
    # Добавим больше текстов для лучшей гистограммы
    more_texts = [(f"Вариант {i}", text1[: 50 * i]) for i in range(1, 20)]
    batch2 = analyzer.analyze_batch(more_texts)

    fig5 = create_entropy_histogram(batch2, nbins=15, show_kde=True)
    fig5.write_html(output_dir / "5_entropy_histogram.html")
    print("   ✓ Сохранено: demo_output/5_entropy_histogram.html")

    # ==== График 6: Скользящая энтропия ====
    print("\n6️⃣  Создание графика скользящей энтропии...")
    long_text = text1 * 10  # Длинный текст
    rolling = analyzer.rolling_entropy(long_text, window_size=20, step_size=5)

    fig6 = create_rolling_entropy_chart(rolling)
    fig6.write_html(output_dir / "6_rolling_entropy.html")
    print("   ✓ Сохранено: demo_output/6_rolling_entropy.html")

    # ==== График 7: Radar сравнение ====
    print("\n7️⃣  Создание радарного графика...")
    text_pushkin = "Мой дядя самых честных правил когда не в шутку занемог"
    text_lermontov = "Белеет парус одинокий в тумане моря голубом что ищет он"
    text_blok = "Ночь улица фонарь аптека бессмысленный и тусклый свет"

    r1 = analyzer.analyze(text_pushkin, "Пушкин")
    r2 = analyzer.analyze(text_lermontov, "Лермонтов")
    r3 = analyzer.analyze(text_blok, "Блок")

    fig7 = create_radar_chart([r1, r2, r3], ["Пушкин", "Лермонтов", "Блок"])
    fig7.write_html(output_dir / "7_radar_comparison.html")
    print("   ✓ Сохранено: demo_output/7_radar_comparison.html")

    # ==== График 8: Heatmap сравнений ====
    print("\n8️⃣  Создание тепловой карты...")
    # Вычислим JS divergence для всех пар
    texts_for_heatmap = [
        ("Пушкин", text_pushkin),
        ("Лермонтов", text_lermontov),
        ("Блок", text_blok),
    ]

    n = len(texts_for_heatmap)
    matrix = [[0.0] * n for _ in range(n)]
    labels = [name for name, _ in texts_for_heatmap]

    for i in range(n):
        for j in range(n):
            if i != j:
                comp = analyzer.compare(texts_for_heatmap[i][1], texts_for_heatmap[j][1])
                matrix[i][j] = comp.js_divergence

    fig8 = create_comparison_heatmap(matrix, labels, "JS Divergence")
    fig8.write_html(output_dir / "8_comparison_heatmap.html")
    print("   ✓ Сохранено: demo_output/8_comparison_heatmap.html")

    # ==== График 9: Dual author ====
    print("\n9️⃣  Создание сравнения двух авторов...")
    pushkin_texts = [(f"Пушкин {i}", text_pushkin[: 20 * i]) for i in range(1, 10)]
    lermontov_texts = [(f"Лермонтов {i}", text_lermontov[: 20 * i]) for i in range(1, 10)]

    batch_pushkin = analyzer.analyze_batch(pushkin_texts)
    batch_lermontov = analyzer.analyze_batch(lermontov_texts)

    fig9 = create_dual_author_comparison(batch_pushkin, batch_lermontov, "Пушкин", "Лермонтов")
    fig9.write_html(output_dir / "9_dual_author.html")
    print("   ✓ Сохранено: demo_output/9_dual_author.html")

    # ==== Итог ====
    print("\n" + "=" * 60)
    print("✅ Все 9 графиков созданы!")
    print("=" * 60)
    print(f"\n📁 Откройте файлы в: {output_dir.absolute()}/")
    print("\n💡 Или откройте все сразу:")
    print(f"   cd {output_dir} && open *.html")
    print("\n🎉 Демонстрация завершена!")


if __name__ == "__main__":
    main()
