#!/usr/bin/env python3
"""
Демонстрация продвинутых статистических тестов.

Показывает максимально объективный анализ различий между авторами.
"""

import numpy as np

from entropy_analysis import TextAnalyzer
from entropy_analysis.core.stats import (
    bootstrap_difference_ci,
    calculate_cohens_d,
    calculate_information_distances,
    compare_groups_statistically,
    permutation_test,
)


def print_section(title):
    """Красивый разделитель."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def example_permutation_test():
    """Пример 1: Permutation Test - самый строгий тест."""
    print_section("Permutation Test - Самый строгий!")

    # Симулируем данные двух авторов
    np.random.seed(42)
    entropies_author1 = np.random.normal(3.8, 0.3, 100)  # Автор 1
    entropies_author2 = np.random.normal(3.6, 0.3, 100)  # Автор 2

    print("\nСравниваем энтропии двух авторов:")
    print(
        f"Автор 1: mean = {np.mean(entropies_author1):.4f}, std = {np.std(entropies_author1):.4f}"
    )
    print(
        f"Автор 2: mean = {np.mean(entropies_author2):.4f}, std = {np.std(entropies_author2):.4f}"
    )

    # Permutation test
    result = permutation_test(entropies_author1, entropies_author2, n_permutations=10000)

    print("\n📊 Результаты Permutation Test:")
    print(f"   Различие: {result.observed_difference:.4f}")
    print(f"   p-value: {result.p_value:.4f}")
    print(f"   Направление: {result.effect_direction}")
    print(f"   Статистически значимо: {'✅ ДА' if result.is_significant else '⚪ НЕТ'}")

    if result.is_significant:
        print("\n💡 Интерпретация:")
        print(f"   С вероятностью {(1 - result.p_value) * 100:.1f}% различие НЕ случайное!")
        print("   Авторы объективно различаются по энтропии.")


def example_effect_size():
    """Пример 2: Effect Size - насколько БОЛЬШОЕ различие."""
    print_section("Effect Size (Cohen's d)")

    # Симулируем разные размеры эффектов
    np.random.seed(42)

    scenarios = [
        ("Малый эффект", np.random.normal(3.5, 0.3, 50), np.random.normal(3.4, 0.3, 50)),
        ("Средний эффект", np.random.normal(3.5, 0.3, 50), np.random.normal(3.2, 0.3, 50)),
        ("Большой эффект", np.random.normal(3.5, 0.3, 50), np.random.normal(3.0, 0.3, 50)),
    ]

    for name, data1, data2 in scenarios:
        result = calculate_cohens_d(data1, data2)

        print(f"\n{name}:")
        print(f"   Cohen's d: {result.cohens_d:.3f}")
        print(f"   Величина: {result.effect_magnitude}")
        print(f"   Различие средних: {result.mean_difference:.4f}")

        if result.effect_magnitude == "large":
            print("   💡 Большое практическое значение!")
        elif result.effect_magnitude == "medium":
            print("   💡 Среднее практическое значение")
        else:
            print("   💡 Малое практическое значение")


def example_bootstrap_ci():
    """Пример 3: Bootstrap CI для различия."""
    print_section("Bootstrap Confidence Interval для различия")

    np.random.seed(42)
    data1 = np.random.normal(14.5, 2.0, 80)  # Perplexity автора 1
    data2 = np.random.normal(13.2, 2.0, 80)  # Perplexity автора 2

    print("\nСравниваем Perplexity:")
    print(f"Автор 1: {np.mean(data1):.2f} ± {np.std(data1):.2f}")
    print(f"Автор 2: {np.mean(data2):.2f} ± {np.std(data2):.2f}")

    result = bootstrap_difference_ci(data1, data2, n_bootstrap=5000)

    print("\n📊 Bootstrap 95% CI:")
    print(f"   Различие: {result.mean_difference:.2f}")
    print(f"   95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    print(f"   Значимо: {'✅ ДА' if result.is_significant else '⚪ НЕТ'}")

    if result.is_significant:
        print("\n💡 Интерпретация:")
        print("   0 НЕ входит в доверительный интервал")
        print("   → Различие статистически значимо с 95% уверенностью")

        if result.mean_difference > 0:
            print(f"   → Автор 1 имеет ВЫШЕ Perplexity на {result.mean_difference:.2f}")
        else:
            print(f"   → Автор 2 имеет ВЫШЕ Perplexity на {abs(result.mean_difference):.2f}")


def example_comprehensive_comparison():
    """Пример 4: Комплексное сравнение (все тесты сразу)."""
    print_section("Комплексное статистическое сравнение")

    np.random.seed(42)
    # Более реалистичные данные
    evenness1 = np.random.beta(20, 2, 120)  # Автор 1: более равномерный
    evenness2 = np.random.beta(15, 3, 100)  # Автор 2: менее равномерный

    print("\nСравниваем Evenness (равномерность):")
    print(f"Автор 1: {np.mean(evenness1):.4f}")
    print(f"Автор 2: {np.mean(evenness2):.4f}")

    # Комплексное сравнение
    result = compare_groups_statistically(
        evenness1, evenness2, n_permutations=5000, n_bootstrap=3000
    )

    print("\n📊 Результаты комплексного анализа:")
    print("\n1. Базовая статистика:")
    print(f"   Автор 1: mean={result.mean1:.4f}, std={result.std1:.4f}, n={result.n1}")
    print(f"   Автор 2: mean={result.mean2:.4f}, std={result.std2:.4f}, n={result.n2}")

    print("\n2. Permutation Test:")
    print(f"   p-value: {result.permutation_test.p_value:.4f}")
    print(f"   Значимо: {'✅ ДА' if result.permutation_test.is_significant else '⚪ НЕТ'}")

    print("\n3. Effect Size:")
    print(f"   Cohen's d: {result.effect_size.cohens_d:.3f}")
    print(f"   Величина: {result.effect_size.effect_magnitude}")

    print("\n4. Bootstrap CI:")
    print(f"   Различие: {result.bootstrap_diff.mean_difference:.4f}")
    print(
        f"   95% CI: [{result.bootstrap_diff.ci_lower:.4f}, {result.bootstrap_diff.ci_upper:.4f}]"
    )

    print("\n5. Классические тесты (для сравнения):")
    print(f"   t-test p-value: {result.t_test_p_value:.4f}")
    print(f"   Mann-Whitney p-value: {result.mann_whitney_p_value:.4f}")

    # Итоговое заключение
    print("\n💡 ИТОГОВОЕ ЗАКЛЮЧЕНИЕ:")

    all_significant = (
        result.permutation_test.is_significant
        and result.bootstrap_diff.is_significant
        and result.t_test_p_value < 0.05
    )

    if all_significant:
        print("   ✅ ВСЕ ТЕСТЫ ПОКАЗЫВАЮТ ЗНАЧИМОЕ РАЗЛИЧИЕ!")
        print("   → Авторы ОБЪЕКТИВНО различаются по Evenness")

        if result.effect_size.effect_magnitude == "large":
            print("   → Различие БОЛЬШОЕ (Cohen's d > 0.8)")
        elif result.effect_size.effect_magnitude == "medium":
            print("   → Различие СРЕДНЕЕ (Cohen's d > 0.5)")
        else:
            print("   → Различие малое, но статистически значимое")
    else:
        print("   ⚪ Нет убедительных доказательств различия")


def example_information_distances():
    """Пример 5: Информационные расстояния между распределениями."""
    print_section("Информационные расстояния")

    # Симулируем распределения букв
    np.random.seed(42)
    dist1 = np.random.dirichlet([1] * 28)  # Автор 1
    dist2 = np.random.dirichlet([1.2] * 28)  # Автор 2 (слегка другое)

    result = calculate_information_distances(dist1, dist2)

    print("\n📊 Различные меры расстояния между распределениями:")
    print(f"\n   Wasserstein (Earth Mover's): {result.wasserstein:.6f}")
    print("   └─ 'Работа' для превращения одного в другое")

    print(f"\n   Hellinger: {result.hellinger:.6f}")
    print("   └─ Геометрическое расстояние (0 = идентичны, 1 = максимально различны)")

    print(f"\n   Bhattacharyya: {result.bhattacharyya:.6f}")
    print("   └─ Связана с вероятностью ошибки классификации")

    print(f"\n   Total Variation: {result.total_variation:.6f}")
    print("   └─ Максимальная разница вероятностей")

    print("\n💡 Интерпретация:")
    if result.hellinger < 0.1:
        print("   Распределения очень похожи (Hellinger < 0.1)")
    elif result.hellinger < 0.3:
        print("   Распределения умеренно различаются")
    else:
        print("   Распределения сильно различаются")


def example_real_world():
    """Пример 6: Реальный пример с текстами."""
    print_section("Реальный пример: Сравнение авторов")

    analyzer = TextAnalyzer.create(alphabet="rus29")

    # Симулируем несколько стихов каждого автора
    texts_pushkin = [
        "Мой дядя самых честных правил когда не в шутку занемог",
        "Унылая пора очей очарованье приятна мне твоя прощальная краса",
        "Я помню чудное мгновенье передо мной явилась ты",
        "Буря мглою небо кроет вихри снежные крутя",
        "В тот год осенняя погода стояла долго на дворе",
    ]

    texts_lermontov = [
        "Белеет парус одинокий в тумане моря голубом",
        "Ночь улица фонарь аптека бессмысленный и тусклый свет",
        "Выхожу один я на дорогу сквозь туман кремнистый путь блестит",
        "Горные вершины спят во тьме ночной",
        "Печально я гляжу на наше поколенье",
    ]

    # Анализируем
    entropies_p = []
    perplexities_p = []

    for text in texts_pushkin:
        result = analyzer.analyze(text)
        entropies_p.append(result.shannon_entropy)
        perplexities_p.append(result.enhanced.perplexity)

    entropies_l = []
    perplexities_l = []

    for text in texts_lermontov:
        result = analyzer.analyze(text)
        entropies_l.append(result.shannon_entropy)
        perplexities_l.append(result.enhanced.perplexity)

    print("\nПушкин:")
    print(f"   Энтропия: {np.mean(entropies_p):.4f} ± {np.std(entropies_p):.4f}")
    print(f"   Perplexity: {np.mean(perplexities_p):.2f} ± {np.std(perplexities_p):.2f}")

    print("\nЛермонтов:")
    print(f"   Энтропия: {np.mean(entropies_l):.4f} ± {np.std(entropies_l):.4f}")
    print(f"   Perplexity: {np.mean(perplexities_l):.2f} ± {np.std(perplexities_l):.2f}")

    # Статистическое сравнение
    print("\n📊 Статистическое сравнение:")

    comp = compare_groups_statistically(
        entropies_p, entropies_l, n_permutations=5000, n_bootstrap=3000
    )

    print("\nЭнтропия:")
    print(f"   p-value: {comp.permutation_test.p_value:.4f}")
    print(f"   Cohen's d: {comp.effect_size.cohens_d:.3f} ({comp.effect_size.effect_magnitude})")
    print(f"   Значимо: {'✅ ДА' if comp.permutation_test.is_significant else '⚪ НЕТ'}")


def main():
    """Запуск всех примеров."""
    print("\n" + "🔬 " * 30)
    print("   ПРОДВИНУТЫЕ СТАТИСТИЧЕСКИЕ ТЕСТЫ")
    print("   Максимальная объективность анализа")
    print("🔬 " * 30 + "\n")

    try:
        example_permutation_test()
        example_effect_size()
        example_bootstrap_ci()
        example_comprehensive_comparison()
        example_information_distances()
        example_real_world()

        print("\n" + "=" * 70)
        print("🎉 ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 70)

        print("\n📚 НОВЫЕ ВОЗМОЖНОСТИ:")
        print("   ✅ Permutation Test - самый строгий, без предположений")
        print("   ✅ Effect Size (Cohen's d) - размер различия")
        print("   ✅ Bootstrap CI - доверительный интервал для различия")
        print("   ✅ Информационные расстояния - Wasserstein, Hellinger и др.")
        print("   ✅ Комплексное сравнение - все тесты сразу")

        print("\n💡 ГДЕ ИСПОЛЬЗОВАТЬ:")
        print("   📊 Веб-интерфейс → вкладка '📊 Статистические тесты'")
        print("   🐍 Python API → from entropy_analysis.core.stats import ...")

        print("\n🎯 ГЛАВНОЕ:")
        print("   Теперь вы можете сказать с НАУЧНОЙ СТРОГОСТЬЮ:")
        print("   • Различаются ли авторы (p-value)")
        print("   • Насколько большое различие (Cohen's d)")
        print("   • С какой уверенностью (95% CI)")
        print()

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
