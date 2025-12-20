#!/usr/bin/env python3
"""
Quick test script for new poetry-specific features.

Usage:
    python test_poetry_features.py

Tests poetry features on sample texts and shows which features
are most informative for Pushkin vs Lermontov.
"""

import sys
from pathlib import Path

# Sample texts for quick testing
PUSHKIN_SAMPLE = """
Я помню чудное мгновенье:
Передо мной явилась ты,
Как мимолетное виденье,
Как гений чистой красоты.

В томленьях грусти безнадежной,
В тревогах шумной суеты,
Звучал мне долго голос нежный
И снились милые черты.
"""

LERMONTOV_SAMPLE = """
Выхожу один я на дорогу;
Сквозь туман кремнистый путь блестит;
Ночь тиха. Пустыня внемлет богу,
И звезда с звездою говорит.

В небесах торжественно и чудно!
Спит земля в сиянье голубом...
Что же мне так больно и так трудно?
Жду ль чего? жалею ли о чём?
"""


def test_individual_features():
    """Test each poetry feature individually on sample texts."""
    try:
        from src.entropy_analysis.core.poetry_features import POETRY_FEATURE_DEFINITIONS
    except ImportError:
        print("❌ Could not import poetry_features module.")
        print("Make sure you're running from the project root.")
        return False
    
    print("=" * 70)
    print("🎭 Testing Poetry Features on Sample Texts")
    print("=" * 70)
    print()
    
    results = []
    
    for feat_def in POETRY_FEATURE_DEFINITIONS:
        name = feat_def["name"]
        extractor = feat_def["extractor"]
        
        pushkin_value = extractor(PUSHKIN_SAMPLE)
        lermontov_value = extractor(LERMONTOV_SAMPLE)
        
        discriminative = "✓" if pushkin_value != lermontov_value else " "
        
        results.append({
            "name": name,
            "pushkin": pushkin_value,
            "lermontov": lermontov_value,
            "discriminative": discriminative,
        })
        
        print(f"{discriminative} {name:25s} | Pushkin: {pushkin_value:20s} | Lermontov: {lermontov_value:20s}")
    
    print()
    print(f"Discriminative features: {sum(1 for r in results if r['discriminative'] == '✓')}/{len(results)}")
    print()
    
    return True


def test_batch_recognition():
    """Test full recognition pipeline with poetry profile."""
    try:
        from src.entropy_analysis.core.recognition_batch import (
            Segment, train_and_run_batch
        )
    except ImportError:
        print("❌ Could not import recognition_batch module.")
        return False
    
    print("=" * 70)
    print("🔬 Testing Batch Recognition with poetry_essential Profile")
    print("=" * 70)
    print()
    
    # Create sample segments
    segments = [
        # Pushkin samples
        Segment(text=PUSHKIN_SAMPLE, label="pushkin", name="pushkin_1"),
        Segment(text="""
Мороз и солнце; день чудесный!
Еще ты дремлешь, друг прелестный —
Пора, красавица, проснись:
Открой сомкнуты негой взоры
Навстречу северной Авроры,
Звездою севера явись!
        """, label="pushkin", name="pushkin_2"),
        Segment(text="""
Я вас любил: любовь еще, быть может,
В душе моей угасла не совсем;
Но пусть она вас больше не тревожит;
Я не хочу печалить вас ничем.
        """, label="pushkin", name="pushkin_3"),
        
        # Lermontov samples
        Segment(text=LERMONTOV_SAMPLE, label="lermontov", name="lermontov_1"),
        Segment(text="""
Утес
Ночевала тучка золотая
На груди утеса-великана;
Утром в путь она умчалась рано,
По лазури весело играя;

Но остался влажный след в морщине
Старого утеса. Одиноко
Он стоит, задумался глубоко,
И тихонько плачет он в пустыне.
        """, label="lermontov", name="lermontov_2"),
        Segment(text="""
Парус
Белеет парус одинокой
В тумане моря голубом!..
Что ищет он в стране далекой?
Что кинул он в краю родном?..

Играют волны — ветер свищет,
И мачта гнется и скрыпит...
Увы! он счастия не ищет
И не от счастия бежит!
        """, label="lermontov", name="lermontov_3"),
    ]
    
    try:
        # Run with poetry_essential profile + automatic lexicons
        result = train_and_run_batch(
            segments=segments,
            feature_profile="poetry_essential",
            smoothing=1e-3,
            error_target=0.05,
            use_author_lexicons=True,  # автоматически извлечёт слова Пушкина/Лермонтова
        )
        
        print("📊 Feature Informativeness (sorted):")
        print()
        for feat in result.recognition.features[:10]:  # top 10
            print(f"  {feat.name:30s} I={feat.informativeness:.4f}  P(e)={feat.error:.4f}")
        
        print()
        print(f"🎯 Best single feature: {result.recognition.best_feature.name}")
        print(f"   Error: {result.recognition.best_feature.error:.4f}")
        
        if result.recognition.best_pair:
            print()
            print(f"🎯 Best pair: {result.recognition.best_pair.names}")
            print(f"   Error: {result.recognition.best_pair.error:.4f}")
        
        print()
        print("🔮 Predictions:")
        print()
        
        correct = 0
        for pred in result.predictions:
            match = "✓" if pred.predicted_label == pred.true_label else "✗"
            confidence = pred.posteriors[pred.predicted_label]
            
            if pred.predicted_label == pred.true_label:
                correct += 1
            
            print(f"  {match} {pred.name:15s} → {pred.predicted_label:12s} (conf: {confidence:.3f})")
            if pred.predicted_label != pred.true_label:
                print(f"      True: {pred.true_label}, Posteriors: {pred.posteriors}")
        
        accuracy = correct / len(result.predictions)
        print()
        print(f"📈 Accuracy: {accuracy:.1%} ({correct}/{len(result.predictions)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during batch recognition: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "POETRY FEATURES TEST SUITE" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Test 1: Individual features
    success1 = test_individual_features()
    
    if not success1:
        print("\n⚠️  Skipping batch test due to import errors.\n")
        return 1
    
    # Test 2: Batch recognition
    success2 = test_batch_recognition()
    
    if success1 and success2:
        print()
        print("=" * 70)
        print("✅ All tests passed!")
        print()
        print("Next steps:")
        print("  1. Try poetry_essential or poetry_full profile in the dashboard")
        print("  2. Load your full Pushkin/Lermontov corpus")
        print("  3. Compare accuracy vs baseline profile")
        print("  4. Check which features have highest informativeness")
        print("=" * 70)
        print()
        return 0
    else:
        print()
        print("❌ Some tests failed. Check errors above.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

