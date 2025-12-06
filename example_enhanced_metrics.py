#!/usr/bin/env python3
"""
Демонстрация улучшенных метрик для энтропийного анализа.

Показывает новые метрики для более полного и точного анализа текстов.
"""

from entropy_analysis import TextAnalyzer


def print_separator(title=""):
    """Красивый разделитель с заголовком."""
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print("=" * 70)
    else:
        print("─" * 70)


def display_enhanced_metrics(result):
    """Показать улучшенные метрики в удобном формате."""
    if not result.enhanced:
        print("⚠ Улучшенные метрики недоступны")
        return

    e = result.enhanced

    print("\n🔬 УЛУЧШЕННЫЕ МЕТРИКИ")
    print_separator()

    print("\n1️⃣  ИНТЕРПРЕТИРУЕМОСТЬ")
    print(f"   Perplexity (Перплексия):          {e.perplexity:.4f}")
    print("   └─ Эффективное число категорий (2^H)")
    print("   └─ Чем выше, тем больше разнообразие")

    print("\n2️⃣  ИСПОЛЬЗОВАНИЕ АЛФАВИТА")
    print(
        f"   Alphabet Utilization:             {e.alphabet_utilization:.4f} ({e.alphabet_utilization * 100:.1f}%)"
    )
    print("   └─ Доля использованных букв алфавита")

    print(f"   Uniqueness Ratio:                 {e.uniqueness_ratio:.4f}")
    print("   └─ Уникальных букв на слово")

    print("\n3️⃣  КОНЦЕНТРАЦИЯ И РАЗНООБРАЗИЕ")
    print(f"   Herfindahl Index (HHI):           {e.herfindahl_index:.4f}")
    print("   └─ Концентрация распределения (выше = меньше разнообразие)")

    print(f"   Evenness (Pielou's J):            {e.evenness:.4f}")
    print("   └─ Равномерность распределения (1 = идеальная)")

    print("\n4️⃣  ОТКЛОНЕНИЕ ОТ ИДЕАЛА")
    print(f"   Uniformity Distance:              {e.uniformity_distance:.4f}")
    print("   └─ Евклидово расстояние от равномерного распределения")

    print(f"   Redundancy:                       {e.redundancy:.4f} ({e.redundancy * 100:.1f}%)")
    print("   └─ Избыточность (1 - H/H_max)")

    print("\n5️⃣  СЕМЕЙСТВО ЭНТРОПИЙ РЕНЬИ")
    print(f"   Rényi H₀ (Hartley):               {e.renyi_0:.4f}")
    print("   └─ log₂(число используемых букв)")

    print(f"   Rényi H₁ (Shannon):               {result.shannon_entropy:.4f}")
    print("   └─ Стандартная энтропия Шеннона")

    print(f"   Rényi H₂ (Collision):             {e.renyi_2:.4f}")
    print("   └─ Вероятность коллизии")

    print(f"   Rényi H∞ (Min-entropy):           {e.renyi_inf:.4f}")
    print("   └─ Энтропия наихудшего случая")


def example_1_single_text():
    """Пример 1: Анализ одного текста с улучшенными метриками."""
    print_separator("ПРИМЕР 1: Полный анализ одного текста")

    analyzer = TextAnalyzer.create(alphabet="rus29")

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

    result = analyzer.analyze(text, source_name="Евгений Онегин")

    # Базовые метрики
    print("\n📊 БАЗОВЫЕ МЕТРИКИ")
    print_separator()
    print(f"   Слов (N):                         {result.n_words}")
    print(f"   Уникальных букв:                  {result.n_unique_letters}")
    print(f"   Shannon Entropy (H):              {result.shannon_entropy:.4f} бит")
    print(f"   Средний ранг (x̄):                {result.mean_rank:.2f}")
    print(f"   Стд. откл. ранга (σ):            {result.std_rank:.2f}")

    # Улучшенные метрики
    display_enhanced_metrics(result)

    # Интерпретация
    print("\n💡 ИНТЕРПРЕТАЦИЯ")
    print_separator()

    if result.enhanced:
        e = result.enhanced

        # Perplexity
        print(f"\n🔹 Перплексия {e.perplexity:.2f} означает, что распределение эквивалентно")
        print(f"   равномерному распределению по {e.perplexity:.0f} буквам")

        # Utilization
        util_pct = e.alphabet_utilization * 100
        if util_pct > 80:
            util_msg = "отлично! используется большая часть алфавита"
        elif util_pct > 60:
            util_msg = "хорошо, используется большинство букв"
        else:
            util_msg = "ограниченное использование алфавита"
        print(f"\n🔹 Использование алфавита {util_pct:.1f}% - {util_msg}")

        # Evenness
        if e.evenness > 0.9:
            even_msg = "очень равномерное"
        elif e.evenness > 0.7:
            even_msg = "умеренно равномерное"
        else:
            even_msg = "неравномерное"
        print(f"\n🔹 Распределение {even_msg} (evenness = {e.evenness:.3f})")

        # Redundancy
        red_pct = e.redundancy * 100
        print(f"\n🔹 Избыточность {red_pct:.1f}% - текст использует {100 - red_pct:.1f}%")
        print("   от максимальной информационной ёмкости")


def example_2_compare_texts():
    """Пример 2: Сравнение разных типов текстов."""
    print_separator("ПРИМЕР 2: Сравнение разных текстов")

    analyzer = TextAnalyzer.create(alphabet="rus29")

    texts = [
        (
            "Поэзия (Пушкин)",
            """
            Мой дядя самых честных правил,
            Когда не в шутку занемог,
            Он уважать себя заставил
            И лучше выдумать не мог.
        """,
        ),
        (
            "Поэзия (Блок)",
            """
            Ночь, улица, фонарь, аптека,
            Бессмысленный и тусклый свет.
            Живи еще хоть четверть века —
            Всё будет так. Исхода нет.
        """,
        ),
        (
            "Проза",
            """
            В начале июля в чрезвычайно жаркое время под вечер
            один молодой человек вышел из своей каморки которую
            нанимал от жильцов в переулке и медленно как бы
            в нерешительности отправился к мосту.
        """,
        ),
    ]

    results = []
    for name, text in texts:
        result = analyzer.analyze(text, source_name=name)
        results.append((name, result))

    # Сравнительная таблица
    print("\n📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print_separator()
    print(f"{'Текст':<20} {'H':>8} {'Perp':>8} {'Even':>8} {'Util%':>8} {'Redun%':>8}")
    print_separator()

    for name, result in results:
        if result.enhanced:
            print(
                f"{name:<20} "
                f"{result.shannon_entropy:>8.4f} "
                f"{result.enhanced.perplexity:>8.2f} "
                f"{result.enhanced.evenness:>8.3f} "
                f"{result.enhanced.alphabet_utilization * 100:>8.1f} "
                f"{result.enhanced.redundancy * 100:>8.1f}"
            )

    # Детальное сравнение
    print("\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ КАЖДОГО ТЕКСТА")
    for name, result in results:
        print(f"\n▶ {name}")
        print("  " + "─" * 66)
        if result.enhanced:
            e = result.enhanced
            print(
                f"  Shannon H: {result.shannon_entropy:.4f} | "
                f"Perplexity: {e.perplexity:.2f} | "
                f"HHI: {e.herfindahl_index:.4f}"
            )
            print(
                f"  Evenness: {e.evenness:.3f} | "
                f"Utilization: {e.alphabet_utilization * 100:.1f}% | "
                f"Redundancy: {e.redundancy * 100:.1f}%"
            )


def example_3_renyi_spectrum():
    """Пример 3: Спектр энтропий Реньи."""
    print_separator("ПРИМЕР 3: Спектр энтропий Реньи")

    analyzer = TextAnalyzer.create(alphabet="rus29")

    text = """
    Унылая пора! Очей очарованье!
    Приятна мне твоя прощальная краса —
    Люблю я пышное природы увяданье,
    В багрец и в золото одетые леса.
    """

    result = analyzer.analyze(text, source_name="Пушкин - Осень")

    print("\n📈 СПЕКТР ЭНТРОПИЙ РЕНЬИ")
    print_separator()
    print("\nЭнтропия Реньи порядка α показывает чувствительность к разным аспектам распределения:")
    print()

    if result.enhanced:
        e = result.enhanced

        print(f"   H₀ (α=0)  Hartley:      {e.renyi_0:.4f} бит")
        print("   └─ Логарифм числа используемых букв")
        print("   └─ Не зависит от частот, только от наличия/отсутствия")
        print()

        print(f"   H₁ (α=1)  Shannon:      {result.shannon_entropy:.4f} бит")
        print("   └─ Стандартная информационная энтропия")
        print("   └─ Оптимальный баланс между редкими и частыми буквами")
        print()

        print(f"   H₂ (α=2)  Collision:    {e.renyi_2:.4f} бит")
        print("   └─ Связана с вероятностью совпадения двух случайных букв")
        print("   └─ Более чувствительна к частым буквам")
        print()

        print(f"   H∞ (α=∞)  Min-entropy:  {e.renyi_inf:.4f} бит")
        print("   └─ Определяется только самой частой буквой")
        print("   └─ Энтропия 'наихудшего случая' для криптографии")
        print()

        print("📊 Монотонность: H₀ ≥ H₁ ≥ H₂ ≥ H∞")
        print(
            f"   Проверка: {e.renyi_0:.3f} ≥ {result.shannon_entropy:.3f} ≥ "
            f"{e.renyi_2:.3f} ≥ {e.renyi_inf:.3f} ✓"
        )


def example_4_practical_insights():
    """Пример 4: Практические выводы из метрик."""
    print_separator("ПРИМЕР 4: Практические выводы")

    analyzer = TextAnalyzer.create(alphabet="rus29")

    # Разные стили текстов
    texts = {
        "Детский": "Мама мыла раму папа помогал маме мыть раму",
        "Обычный": "В лесу родилась ёлочка в лесу она росла зимой и летом стройная зелёная была",
        "Сложный": "Экзистенциальная феноменология рассматривает онтологические аспекты субъективности",
    }

    print("\n🎯 АНАЛИЗ ТЕКСТОВ РАЗНОЙ СЛОЖНОСТИ")
    print_separator()

    for style, text in texts.items():
        result = analyzer.analyze(text, source_name=style)
        print(f"\n▶ {style} текст")
        print("  " + "─" * 66)
        print(f"  Слов: {result.n_words} | Уникальных букв: {result.n_unique_letters}")

        if result.enhanced:
            e = result.enhanced

            # Оценка сложности
            complexity_score = (
                result.shannon_entropy * 0.3
                + e.perplexity * 0.2
                + e.alphabet_utilization * 5
                + e.evenness * 3
            )

            print(
                f"  Перплексия: {e.perplexity:.2f} | "
                f"Использование алфавита: {e.alphabet_utilization * 100:.1f}%"
            )
            print(f"  Равномерность: {e.evenness:.3f} | Оценка сложности: {complexity_score:.2f}")

            # Интерпретация
            if complexity_score < 5:
                level = "низкая (простой текст)"
            elif complexity_score < 8:
                level = "средняя (обычный текст)"
            else:
                level = "высокая (сложный текст)"

            print(f"  └─ Сложность: {level}")


def main():
    """Запуск всех примеров."""
    print("\n" + "🔬 " * 30)
    print("   УЛУЧШЕННЫЕ МЕТРИКИ ЭНТРОПИЙНОГО АНАЛИЗА")
    print("🔬 " * 30 + "\n")

    try:
        example_1_single_text()
        example_2_compare_texts()
        example_3_renyi_spectrum()
        example_4_practical_insights()

        print("\n" + "=" * 70)
        print("🎉 ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 70)

        print("\n📚 НОВЫЕ МЕТРИКИ:")
        print("   ✓ Perplexity - интуитивная мера разнообразия")
        print("   ✓ Alphabet Utilization - процент использования алфавита")
        print("   ✓ Herfindahl Index - концентрация распределения")
        print("   ✓ Evenness (Pielou's J) - равномерность распределения")
        print("   ✓ Uniformity Distance - отклонение от идеального")
        print("   ✓ Uniqueness Ratio - уникальных букв на слово")
        print("   ✓ Redundancy - избыточность текста")
        print("   ✓ Rényi Entropy (H₀, H₁, H₂, H∞) - спектр энтропий")

        print("\n💡 ПРЕИМУЩЕСТВА:")
        print("   • Более полная картина распределения")
        print("   • Лучшая интерпретируемость")
        print("   • Разные аспекты разнообразия и концентрации")
        print("   • Подходит для сравнения текстов разной природы")
        print()

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
