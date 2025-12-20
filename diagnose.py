#!/usr/bin/env python3
"""
Диагностика проблем с распознаванием.
Показывает, почему точность низкая и какие признаки не работают.
"""

from pathlib import Path
from src.entropy_analysis.core.recognition_batch import (
    Segment, train_and_run_batch
)


def load_files(directory="science"):
    """Загружает все .txt файлы как авторов."""
    segments = []
    dir_path = Path(directory)
    
    for file_path in dir_path.glob("*_filtered.txt"):
        author = file_path.stem  # имя без расширения
        text = file_path.read_text(encoding="utf-8")
        parts = [p.strip() for p in text.split("======") if p.strip()]
        
        for idx, part in enumerate(parts, 1):
            if len(part) > 50:  # только значимые фрагменты
                segments.append(
                    Segment(text=part, label=author, name=f"{author}_{idx}")
                )
    
    return segments


def diagnose():
    """Диагностика проблем."""
    print("=" * 70)
    print("🔍 ДИАГНОСТИКА РАСПОЗНАВАНИЯ")
    print("=" * 70)
    print()
    
    # Загружаем данные
    segments = load_files()
    
    if not segments:
        print("❌ Файлы не найдены в science/")
        print("   Ожидаются: *_filtered.txt")
        return
    
    # Статистика по авторам
    from collections import Counter
    authors = Counter(s.label for s in segments)
    
    print(f"📚 Загружено сегментов: {len(segments)}")
    print(f"👥 Авторы:")
    for author, count in authors.most_common():
        avg_len = sum(len(s.text) for s in segments if s.label == author) / count
        print(f"   {author:25s} {count:4d} сегментов, средняя длина: {avg_len:.0f} символов")
    print()
    
    if len(authors) < 2:
        print("❌ Нужно минимум 2 автора!")
        return
    
    # Проверяем разные профили
    profiles = [
        ("baseline", False),
        ("poetry_essential", True),
        ("poetry_full", True),
    ]
    
    results = []
    
    for profile, use_lex in profiles:
        print(f"\n{'='*70}")
        print(f"📊 Тестируем профиль: {profile} (lexicons={use_lex})")
        print(f"{'='*70}\n")
        
        try:
            result = train_and_run_batch(
                segments=segments,
                feature_profile=profile,
                use_author_lexicons=use_lex,
                smoothing=1e-3,
                error_target=0.05,
            )
            
            # Точность
            correct = sum(1 for p in result.predictions if p.predicted_label == p.true_label)
            accuracy = correct / len(result.predictions)
            
            # Топ-5 признаков
            top5 = result.recognition.features[:5]
            
            print(f"✅ Точность: {accuracy:.1%} ({correct}/{len(result.predictions)})")
            print(f"📉 Мин. ошибка: {result.recognition.min_error:.4f}")
            print(f"🔢 Использовано признаков: {len(result.used_features)}")
            print()
            print(f"🔝 Топ-5 признаков:")
            for feat in top5:
                print(f"   {feat.name:30s} I={feat.informativeness:.4f}  P(e)={feat.error:.4f}")
            
            # Ошибки по авторам
            from collections import defaultdict
            errors_by_author = defaultdict(list)
            for pred in result.predictions:
                if pred.predicted_label != pred.true_label:
                    errors_by_author[pred.true_label].append(
                        (pred.name, pred.predicted_label, pred.posteriors[pred.predicted_label])
                    )
            
            if errors_by_author:
                print()
                print("❌ Ошибки по авторам:")
                for author in sorted(errors_by_author.keys()):
                    errs = errors_by_author[author]
                    print(f"\n   {author} ({len(errs)} ошибок):")
                    for name, pred, conf in errs[:5]:  # показываем первые 5
                        print(f"      {name:20s} → {pred:20s} (уверенность: {conf:.3f})")
            
            results.append({
                "profile": profile,
                "accuracy": accuracy,
                "min_error": result.recognition.min_error,
                "num_features": len(result.used_features),
                "top_feature": top5[0].name if top5 else None,
                "top_info": top5[0].informativeness if top5 else 0,
            })
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    # Итоговое сравнение
    print("\n" + "="*70)
    print("📈 СРАВНЕНИЕ ПРОФИЛЕЙ")
    print("="*70)
    print()
    print(f"{'Профиль':<25} {'Точность':>10} {'Признаков':>12} {'Топ-признак (I)':<35}")
    print("-"*70)
    
    for r in results:
        info_str = f"{r['top_feature']} ({r['top_info']:.3f})" if r['top_feature'] else "—"
        print(f"{r['profile']:<25} {r['accuracy']:>9.1%} {r['num_features']:>12} {info_str:<35}")
    
    # Рекомендации
    print("\n" + "="*70)
    print("💡 РЕКОМЕНДАЦИИ")
    print("="*70)
    
    if results:
        best = max(results, key=lambda x: x['accuracy'])
        print(f"\n🏆 Лучший профиль: {best['profile']} ({best['accuracy']:.1%})")
        
        if best['accuracy'] < 0.8:
            print("\n⚠️  Точность низкая (<80%). Возможные причины:")
            print("   1. Авторы слишком похожи по стилю")
            print("   2. Мало данных для обучения (нужно >50 сегментов на автора)")
            print("   3. Признаки не подходят для этих авторов")
            print("   4. Плохое качество текстов (опечатки, OCR-ошибки)")
        
        if best['top_info'] < 0.1:
            print("\n⚠️  Информативность признаков низкая (<0.1):")
            print("   → Признаки плохо различают авторов")
            print("   → Попробуйте добавить больше данных")
            print("   → Или используйте другие признаки")
    
    print()


if __name__ == "__main__":
    diagnose()

