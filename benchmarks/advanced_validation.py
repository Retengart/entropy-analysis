#!/usr/bin/env python3
"""
Advanced ML validation with full feature extraction and model comparison.

This script performs comprehensive machine learning validation:
- Extracts ALL available entropy features
- Trains Random Forest and CatBoost models
- Performs hyperparameter optimization
- Provides detailed evaluation and feature importance analysis

Usage:
    python benchmarks/advanced_validation.py \
        --dataset benchmarks/datasets/pushkin_vs_lermontov_full.json \
        --output benchmarks/results/advanced_validation_report.md
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not installed. Install: pip install catboost")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from entropy_analysis.core.analyze import TextAnalyzer


# ============================================================================
# Feature Extraction
# ============================================================================

class AdvancedFeatureExtractor:
    """Extract ALL available entropy and linguistic features."""
    
    def __init__(self):
        self.analyzer = TextAnalyzer.create(alphabet="rus29")
        self.feature_names = []
        
    def extract_all_features(self, text: str) -> np.ndarray:
        """Extract comprehensive feature vector from text.
        
        Returns array of ~20+ features covering:
        - Basic entropy metrics
        - Enhanced metrics
        - Advanced Phase 2 metrics
        - Lexical richness
        - Complexity measures
        """
        result = self.analyzer.analyze(
            text,
            include_bootstrap=False,  # Skip for speed
            include_complexity=True,
            include_advanced_metrics=True,
        )
        
        features = []
        feature_names = []
        
        # === BASIC METRICS ===
        if result.shannon_entropy is not None:
            features.append(result.shannon_entropy)
            feature_names.append('shannon_entropy')
        else:
            features.append(0.0)
            feature_names.append('shannon_entropy')
            
        if result.mean_rank is not None:
            features.append(result.mean_rank)
            feature_names.append('mean_rank')
        else:
            features.append(0.0)
            feature_names.append('mean_rank')
            
        if result.std_rank is not None:
            features.append(result.std_rank)
            feature_names.append('std_rank')
        else:
            features.append(0.0)
            feature_names.append('std_rank')
        
        # === ENHANCED METRICS ===
        if result.enhanced:
            features.extend([
                result.enhanced.perplexity,
                result.enhanced.alphabet_utilization,
                result.enhanced.evenness,
                result.enhanced.herfindahl_index,
                result.enhanced.uniformity_distance,
                result.enhanced.uniqueness_ratio,
                result.enhanced.redundancy,
                result.enhanced.renyi_0,
                result.enhanced.renyi_2,
                result.enhanced.renyi_inf,
            ])
            feature_names.extend([
                'perplexity',
                'alphabet_utilization',
                'evenness',
                'herfindahl_index',
                'uniformity_distance',
                'uniqueness_ratio',
                'redundancy',
                'renyi_0',
                'renyi_2',
                'renyi_inf',
            ])
        else:
            features.extend([0.0] * 10)
            feature_names.extend([
                'perplexity', 'alphabet_utilization', 'evenness',
                'herfindahl_index', 'uniformity_distance', 'uniqueness_ratio',
                'redundancy', 'renyi_0', 'renyi_2', 'renyi_inf',
            ])
        
        # === DIVERSITY INDICES ===
        if result.simpson_index is not None:
            features.append(result.simpson_index)
            feature_names.append('simpson_index')
        else:
            features.append(0.0)
            feature_names.append('simpson_index')
            
        if result.gini_simpson_index is not None:
            features.append(result.gini_simpson_index)
            feature_names.append('gini_simpson_index')
        else:
            features.append(0.0)
            feature_names.append('gini_simpson_index')
        
        # === ZIPF ANALYSIS ===
        if result.zipf_alpha is not None:
            features.append(result.zipf_alpha)
            feature_names.append('zipf_alpha')
        else:
            features.append(0.0)
            feature_names.append('zipf_alpha')
        
        # === COMPLEXITY ===
        if result.compression_ratio is not None:
            features.append(result.compression_ratio)
            feature_names.append('compression_ratio')
        else:
            features.append(0.0)
            feature_names.append('compression_ratio')
        
        # === LEXICAL RICHNESS ===
        if result.yules_k is not None:
            features.append(result.yules_k)
            feature_names.append('yules_k')
        else:
            features.append(0.0)
            feature_names.append('yules_k')
            
        if result.mtld is not None:
            features.append(result.mtld)
            feature_names.append('mtld')
        else:
            features.append(0.0)
            feature_names.append('mtld')
            
        if result.mattr is not None:
            features.append(result.mattr)
            feature_names.append('mattr')
        else:
            features.append(0.0)
            feature_names.append('mattr')
            
        if result.hdd is not None:
            features.append(result.hdd)
            feature_names.append('hdd')
        else:
            features.append(0.0)
            feature_names.append('hdd')
        
        # === N-GRAM ENTROPIES ===
        if result.bigram_entropy is not None:
            features.append(result.bigram_entropy)
            feature_names.append('bigram_entropy')
        else:
            features.append(0.0)
            feature_names.append('bigram_entropy')
            
        if result.letter_bigram_entropy is not None:
            features.append(result.letter_bigram_entropy)
            feature_names.append('letter_bigram_entropy')
        else:
            features.append(0.0)
            feature_names.append('letter_bigram_entropy')
            
        if result.letter_trigram_entropy is not None:
            features.append(result.letter_trigram_entropy)
            feature_names.append('letter_trigram_entropy')
        else:
            features.append(0.0)
            feature_names.append('letter_trigram_entropy')
        
        # === BURSTINESS ===
        if result.burstiness is not None:
            features.append(result.burstiness)
            feature_names.append('burstiness')
        else:
            features.append(0.0)
            feature_names.append('burstiness')
        
        # === PHASE 2 METRICS ===
        if result.hurst_exponent is not None:
            features.append(result.hurst_exponent)
            feature_names.append('hurst_exponent')
        else:
            features.append(0.0)
            feature_names.append('hurst_exponent')
        
        # Store feature names on first call
        if not self.feature_names:
            self.feature_names = feature_names
        
        return np.array(features)
    
    def extract_features_batch(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract features from multiple texts.
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
        """
        X = []
        for text in texts:
            features = self.extract_all_features(text)
            X.append(features)
        
        return np.array(X), self.feature_names


# ============================================================================
# Model Training and Evaluation
# ============================================================================

def train_random_forest(X_train, y_train, cv=5, verbose=True):
    """Train Random Forest with hyperparameter optimization."""
    
    if verbose:
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
    
    # Parameter grid - reduced to fight overfitting
    param_grid = {
        'n_estimators': [50, 100, 150],  # Reduced from [100, 200, 300]
        'max_depth': [5, 8, 10],  # Reduced from [10, 20, None]
        'min_samples_split': [5, 10, 20],  # Increased from [2, 5, 10]
        'min_samples_leaf': [4, 8, 10],  # Increased from [1, 2, 4]
        'max_features': ['sqrt', 'log2'],
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )
    
    if verbose:
        print("Running GridSearchCV...")
    
    grid_search.fit(X_train, y_train)
    
    if verbose:
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_catboost(X_train, y_train, cv=5, verbose=True):
    """Train CatBoost with hyperparameter optimization."""
    
    if not CATBOOST_AVAILABLE:
        print("❌ CatBoost not available. Skipping.")
        return None, None, None
    
    if verbose:
        print("\n" + "="*70)
        print("TRAINING CATBOOST")
        print("="*70)
    
    # Parameter grid - reduced to fight overfitting
    param_grid = {
        'iterations': [50, 100, 200],  # Reduced from [100, 200, 500]
        'depth': [4, 5, 6],  # Reduced from [4, 6, 8]
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [3, 5, 10],  # Increased regularization
    }
    
    cb = CatBoostClassifier(
        random_state=42,
        verbose=0,
        early_stopping_rounds=50,
        auto_class_weights='Balanced',  # Handle any remaining imbalance
    )
    
    # Grid search
    grid_search = GridSearchCV(
        cb,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )
    
    if verbose:
        print("Running GridSearchCV...")
    
    grid_search.fit(X_train, y_train)
    
    if verbose:
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_names):
    """Comprehensive model evaluation."""
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities (for ROC)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, average='weighted'
    )
    
    # Check for overfitting
    overfit_gap = train_acc - test_acc
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_test_proba)
    except:
        roc_auc = None
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
    else:
        feature_importance = []
    
    results = {
        'model_name': model_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'overfit_gap': overfit_gap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'feature_importance': feature_importance[:10],  # Top 10
    }
    
    return results


# ============================================================================
# Main Pipeline
# ============================================================================

def load_dataset(path: Path):
    """Load dataset from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data['texts']]
    labels = [item['author_id'] for item in data['texts']]
    author_names = data['authors']
    
    return texts, labels, author_names, data['metadata']


def run_advanced_validation(dataset_path: Path, output_path: Path):
    """Run complete advanced validation pipeline."""
    
    print("\n" + "="*70)
    print("ADVANCED ML VALIDATION PIPELINE")
    print("="*70)
    
    # Load data
    print(f"\n📚 Loading dataset: {dataset_path}")
    texts, labels, author_names, metadata = load_dataset(dataset_path)
    print(f"✅ Loaded {len(texts)} texts from {len(author_names)} authors")
    print(f"   Distribution: {dict(zip(author_names, [labels.count(i) for i in range(len(author_names))]))}")
    
    # Extract features
    print("\n🔬 Extracting ALL features...")
    extractor = AdvancedFeatureExtractor()
    X, feature_names = extractor.extract_features_batch(texts)
    y = np.array(labels)
    
    print(f"✅ Extracted {X.shape[1]} features:")
    for i, name in enumerate(feature_names[:10]):
        print(f"   {i+1}. {name}")
    if len(feature_names) > 10:
        print(f"   ... and {len(feature_names) - 10} more")
    
    # Train/test split
    print("\n📊 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Normalize features
    print("\n⚙️  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    results = {}
    
    # Random Forest
    rf_model, rf_params, rf_cv_score = train_random_forest(
        X_train_scaled, y_train, cv=5, verbose=True
    )
    if rf_model:
        results['RandomForest'] = evaluate_model(
            rf_model, X_train_scaled, X_test_scaled, y_train, y_test,
            "Random Forest", feature_names
        )
        results['RandomForest']['best_params'] = rf_params
        results['RandomForest']['cv_score'] = rf_cv_score
    
    # CatBoost
    cb_model, cb_params, cb_cv_score = train_catboost(
        X_train_scaled, y_train, cv=5, verbose=True
    )
    if cb_model:
        results['CatBoost'] = evaluate_model(
            cb_model, X_train_scaled, X_test_scaled, y_train, y_test,
            "CatBoost", feature_names
        )
        results['CatBoost']['best_params'] = cb_params
        results['CatBoost']['cv_score'] = cb_cv_score
    
    # Print summary
    print_results_summary(results, author_names)
    
    # Generate report
    generate_report(results, metadata, author_names, feature_names, output_path)
    
    return results


def print_results_summary(results: Dict, author_names: List[str]):
    """Print comparison summary."""
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<20} {'Test Acc':>10} {'Train Acc':>10} {'Overfit':>10} {'F1':>10}")
    print("-"*70)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} "
              f"{result['test_accuracy']:>9.2%} "
              f"{result['train_accuracy']:>9.2%} "
              f"{result['overfit_gap']:>9.2%} "
              f"{result['f1']:>9.4f}")
    
    # Winner
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print("="*70)
    print(f"🏆 WINNER: {best_model[0]} ({best_model[1]['test_accuracy']:.2%} accuracy)")
    
    # Overfitting warning
    for model_name, result in results.items():
        if result['overfit_gap'] > 0.10:
            print(f"⚠️  WARNING: {model_name} shows signs of overfitting (gap: {result['overfit_gap']:.2%})")


def generate_report(results, metadata, author_names, feature_names, output_path):
    """Generate markdown report."""
    
    timestamp = datetime.now().isoformat()
    
    report = f"""# Advanced ML Validation Report

**Generated:** {timestamp}

## Dataset

- **Authors:** {len(author_names)} ({', '.join(author_names)})
- **Total texts:** {metadata['n_texts']}
- **Features extracted:** {len(feature_names)}
- **Train/Test split:** 80/20 stratified

### Distribution:
"""
    
    for author, count in metadata['texts_per_author'].items():
        report += f"- **{author}:** {count} texts\n"
    
    report += f"""
## Features Extracted ({len(feature_names)} total)

### Top Features Used:
"""
    
    for i, feat in enumerate(feature_names[:15], 1):
        report += f"{i}. {feat}\n"
    
    report += "\n---\n\n## Results\n\n"
    
    # Comparison table
    report += "| Model | Test Acc | Train Acc | Overfit Gap | Precision | Recall | F1 | ROC-AUC |\n"
    report += "|-------|----------|-----------|-------------|-----------|--------|----|---------|\n"
    
    for model_name, result in results.items():
        roc_str = f"{result['roc_auc']:.4f}" if result['roc_auc'] else "N/A"
        report += (f"| **{model_name}** | "
                  f"**{result['test_accuracy']:.4f}** | "
                  f"{result['train_accuracy']:.4f} | "
                  f"{result['overfit_gap']:.4f} | "
                  f"{result['precision']:.4f} | "
                  f"{result['recall']:.4f} | "
                  f"{result['f1']:.4f} | "
                  f"{roc_str} |\n")
    
    # Winner
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    diff = best_model[1]['test_accuracy'] - min(r['test_accuracy'] for r in results.values())
    
    report += f"\n### Winner\n\n"
    report += f"🏆 **{best_model[0]}** with {best_model[1]['test_accuracy']:.2%} accuracy\n\n"
    
    if diff > 0.02:
        report += f"✅ Significantly better than other model (+{diff:.2%})\n\n"
    else:
        report += f"⚪ Comparable performance (diff: {diff:.2%})\n\n"
    
    # Detailed results for each model
    for model_name, result in results.items():
        report += f"\n---\n\n## {model_name} - Detailed Results\n\n"
        
        # Hyperparameters
        if 'best_params' in result:
            report += "### Best Hyperparameters:\n\n```python\n"
            for param, value in result['best_params'].items():
                report += f"{param}: {value}\n"
            report += "```\n\n"
        
        # Metrics
        report += f"- **Test Accuracy:** {result['test_accuracy']:.4f}\n"
        report += f"- **Train Accuracy:** {result['train_accuracy']:.4f}\n"
        report += f"- **CV Score:** {result.get('cv_score', 'N/A'):.4f}\n"
        report += f"- **Precision:** {result['precision']:.4f}\n"
        report += f"- **Recall:** {result['recall']:.4f}\n"
        report += f"- **F1-Score:** {result['f1']:.4f}\n\n"
        
        # Overfitting check
        if result['overfit_gap'] > 0.10:
            report += f"⚠️ **WARNING: Possible overfitting** (gap: {result['overfit_gap']:.2%})\n\n"
        elif result['overfit_gap'] > 0.05:
            report += f"⚪ **Minor overfitting** (gap: {result['overfit_gap']:.2%})\n\n"
        else:
            report += f"✅ **No significant overfitting** (gap: {result['overfit_gap']:.2%})\n\n"
        
        # Confusion matrix
        report += "### Confusion Matrix:\n\n```\n"
        report += str(result['confusion_matrix'])
        report += "\n```\n\n"
        
        # Feature importance
        if result['feature_importance']:
            report += "### Top 10 Most Important Features:\n\n"
            for i, (feat, importance) in enumerate(result['feature_importance'], 1):
                report += f"{i}. **{feat}**: {importance:.4f}\n"
            report += "\n"
    
    # Conclusions
    report += "\n---\n\n## Conclusions\n\n"
    
    best_acc = best_model[1]['test_accuracy']
    if best_acc >= 0.90:
        report += "✅ **Excellent performance** (≥90% accuracy)\n\n"
    elif best_acc >= 0.80:
        report += "✅ **Good performance** (80-90% accuracy)\n\n"
    elif best_acc >= 0.70:
        report += "⚪ **Moderate performance** (70-80% accuracy)\n\n"
    else:
        report += "❌ **Limited performance** (<70% accuracy)\n\n"
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced ML validation with full features and model comparison"
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to dataset JSON file',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmarks/results/advanced_validation_report.md'),
        help='Output report path',
    )
    
    args = parser.parse_args()
    
    if not args.dataset.exists():
        print(f"❌ Dataset not found: {args.dataset}")
        return 1
    
    try:
        run_advanced_validation(args.dataset, args.output)
        print("\n✅ Validation complete!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

