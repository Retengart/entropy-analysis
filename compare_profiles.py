#!/usr/bin/env python3
"""
Compare recognition accuracy across different feature profiles.

Usage:
    python compare_profiles.py

Compares baseline, compact, poetry_essential, and poetry_full profiles
on the full Pushkin/Lermontov corpus.
"""

import sys
from pathlib import Path
from src.entropy_analysis.core.recognition_batch import (
    Segment, train_and_run_batch
)


def load_corpus(file_path: Path, author: str, separator: str = "======") -> list[Segment]:
    """Load and segment a poetry corpus file."""
    text = file_path.read_text(encoding="utf-8")
    segments = []
    
    parts = [p.strip() for p in text.split(separator) if p.strip()]
    
    for idx, part in enumerate(parts):
        if len(part) > 30:  # skip very short fragments
            segments.append(
                Segment(
                    text=part,
                    label=author,
                    name=f"{author}_{idx+1}"
                )
            )
    
    return segments


def evaluate_profile(segments: list[Segment], profile_name: str):
    """Run recognition with a specific profile and return metrics."""
    print(f"\n{'='*70}")
    print(f"📊 Testing profile: {profile_name}")
    print(f"{'='*70}\n")
    
    try:
        result = train_and_run_batch(
            segments=segments,
            feature_profile=profile_name,
            smoothing=1e-3,
            error_target=0.05,
        )
        
        # Calculate accuracy
        correct = sum(1 for p in result.predictions if p.predicted_label == p.true_label)
        accuracy = correct / len(result.predictions)
        
        # Average confidence
        avg_confidence = sum(
            p.posteriors[p.predicted_label] for p in result.predictions
        ) / len(result.predictions)
        
        # Get top features
        top_features = result.recognition.features[:5]
        
        print(f"✅ Accuracy: {accuracy:.1%} ({correct}/{len(result.predictions)})")
        print(f"🎯 Avg confidence: {avg_confidence:.3f}")
        print(f"📉 Min error: {result.recognition.min_error:.4f}")
        print(f"\n🔝 Top 5 features by informativeness:")
        for feat in top_features:
            print(f"   {feat.name:25s} I={feat.informativeness:.4f}  P(e)={feat.error:.4f}")
        
        if result.recognition.best_feature:
            print(f"\n🎯 Best single: {result.recognition.best_feature.name} (P(e)={result.recognition.best_feature.error:.4f})")
        
        if result.recognition.best_pair:
            print(f"🎯 Best pair: {result.recognition.best_pair.names} (P(e)={result.recognition.best_pair.error:.4f})")
        
        # Show confusion
        from collections import defaultdict
        confusion = defaultdict(lambda: defaultdict(int))
        for pred in result.predictions:
            confusion[pred.true_label][pred.predicted_label] += 1
        
        print(f"\n📊 Confusion matrix:")
        labels = sorted(confusion.keys())
        for true_label in labels:
            print(f"   {true_label:12s} →", end="")
            for pred_label in labels:
                count = confusion[true_label][pred_label]
                print(f" {pred_label}:{count:3d}", end="")
            print()
        
        return {
            "profile": profile_name,
            "accuracy": accuracy,
            "confidence": avg_confidence,
            "min_error": result.recognition.min_error,
            "num_features": len(result.used_features),
            "top_feature": top_features[0].name if top_features else None,
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main comparison."""
    print("\n" + "="*70)
    print("🎭 FEATURE PROFILE COMPARISON: Pushkin vs Lermontov")
    print("="*70)
    
    # Load corpus
    pushkin_file = Path("science/pushkin_filtered.txt")
    lermontov_file = Path("science/lermontov_filtered.txt")
    
    if not pushkin_file.exists() or not lermontov_file.exists():
        print(f"\n❌ Corpus files not found!")
        print(f"   Expected: {pushkin_file.absolute()}")
        print(f"            {lermontov_file.absolute()}")
        return 1
    
    print(f"\n📚 Loading corpus...")
    pushkin_segments = load_corpus(pushkin_file, "pushkin")
    lermontov_segments = load_corpus(lermontov_file, "lermontov")
    
    all_segments = pushkin_segments + lermontov_segments
    
    print(f"   Pushkin: {len(pushkin_segments)} segments")
    print(f"   Lermontov: {len(lermontov_segments)} segments")
    print(f"   Total: {len(all_segments)} segments")
    
    # Test each profile
    profiles = [
        "baseline",
        "compact",
        "poetry_essential",
        "poetry_full",
    ]
    
    results = []
    for profile in profiles:
        result = evaluate_profile(all_segments, profile)
        if result:
            results.append(result)
    
    # Summary table
    print("\n" + "="*70)
    print("📈 SUMMARY")
    print("="*70)
    print()
    print(f"{'Profile':<20} {'Accuracy':>10} {'Confidence':>12} {'Features':>10} {'Best Feature':<25}")
    print("-"*70)
    
    for r in results:
        print(f"{r['profile']:<20} {r['accuracy']:>9.1%} {r['confidence']:>12.3f} {r['num_features']:>10} {r['top_feature']:<25}")
    
    print()
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['accuracy'])
        print(f"🏆 Best profile: {best['profile']} ({best['accuracy']:.1%} accuracy)")
        
        baseline = next((r for r in results if r['profile'] == 'baseline'), None)
        if baseline and best['profile'] != 'baseline':
            improvement = (best['accuracy'] - baseline['accuracy']) * 100
            print(f"   Improvement over baseline: +{improvement:.1f} percentage points")
    
    print()
    print("="*70)
    print("✅ Comparison complete!")
    print()
    print("💡 Recommendation:")
    print("   Use 'poetry_essential' for best balance of accuracy and interpretability.")
    print("   Use 'poetry_full' for maximum accuracy (if validated).")
    print("="*70)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

