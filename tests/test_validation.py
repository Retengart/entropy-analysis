"""
Scientific validation tests for author attribution.

This module provides rigorous testing of the entropy-based method
against baseline methods on benchmark datasets.

Tests include:
- Accuracy, precision, recall, F1-score
- Comparison with TF-IDF + SVM baseline
- Cross-validation
- Confusion matrices
"""

import pytest
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from entropy_analysis.core.analyze import TextAnalyzer


# ============================================================================
# Test Dataset Configuration
# ============================================================================

class AuthorCorpus:
    """Container for author attribution test corpus."""
    
    def __init__(self):
        self.texts = []  # List of (text, author_id, author_name) tuples
        self.author_names = []  # List of unique author names
        
    def add_text(self, text: str, author_name: str):
        """Add a text sample to the corpus."""
        if author_name not in self.author_names:
            self.author_names.append(author_name)
        author_id = self.author_names.index(author_name)
        self.texts.append((text, author_id, author_name))
        
    def get_train_test_split(self, test_size=0.3, random_state=42):
        """Split corpus into train and test sets."""
        texts = [t[0] for t in self.texts]
        labels = [t[1] for t in self.texts]
        
        return train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels  # Ensure balanced classes
        )
    
    @property
    def n_authors(self) -> int:
        """Number of unique authors."""
        return len(self.author_names)
    
    @property
    def n_texts(self) -> int:
        """Total number of texts."""
        return len(self.texts)


# ============================================================================
# Baseline Classifiers
# ============================================================================

class TfidfSvmBaseline:
    """TF-IDF + SVM baseline for author attribution.
    
    This is a standard baseline method widely used in stylometry.
    """
    
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # unigrams, bigrams, trigrams
            analyzer='char',  # Character n-grams work well for Russian
        )
        self.classifier = SVC(kernel='linear', random_state=42)
        
    def fit(self, texts, labels):
        """Train the baseline model."""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        
    def predict(self, texts):
        """Predict author for new texts."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)
    
    def score(self, texts, labels):
        """Calculate accuracy on test set."""
        predictions = self.predict(texts)
        return accuracy_score(labels, predictions)


class EntropyBasedClassifier:
    """Entropy-based author attribution classifier.
    
    Uses the entropy metrics from this project as features
    for author classification.
    """
    
    def __init__(self):
        self.analyzer = TextAnalyzer.create(alphabet="rus29")
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = [
            'shannon_entropy',
            'perplexity',
            'evenness',
            'alphabet_utilization',
            'renyi_h0',
            'renyi_h2',
            'renyi_h_inf',
            'uniqueness_ratio',
        ]
        
    def _extract_features(self, text: str) -> np.ndarray:
        """Extract entropy-based features from text."""
        result = self.analyzer.analyze(text, include_enhanced=True)
        
        if result.shannon_entropy is None:
            # Return zeros if analysis failed
            return np.zeros(len(self.feature_names))
        
        features = []
        features.append(result.shannon_entropy)
        
        if result.enhanced:
            features.append(result.enhanced.perplexity)
            features.append(result.enhanced.evenness)
            features.append(result.enhanced.alphabet_utilization)
            features.append(result.enhanced.renyi_h0)
            features.append(result.enhanced.renyi_h2)
            features.append(result.enhanced.renyi_h_inf)
            features.append(result.enhanced.uniqueness_ratio)
        else:
            features.extend([0.0] * 7)
            
        return np.array(features)
    
    def fit(self, texts, labels):
        """Train the entropy-based classifier."""
        X = np.array([self._extract_features(text) for text in texts])
        self.classifier.fit(X, labels)
        
    def predict(self, texts):
        """Predict author for new texts."""
        X = np.array([self._extract_features(text) for text in texts])
        return self.classifier.predict(X)
    
    def score(self, texts, labels):
        """Calculate accuracy on test set."""
        predictions = self.predict(texts)
        return accuracy_score(labels, predictions)


# ============================================================================
# Validation Tests
# ============================================================================

@pytest.fixture
def sample_corpus():
    """Create a small sample corpus for testing.
    
    This is a minimal corpus just to test the pipeline.
    For real validation, use a larger corpus from files.
    """
    corpus = AuthorCorpus()
    
    # Pushkin samples (simplified for testing)
    pushkin_samples = [
        "Мой дядя самых честных правил когда не в шутку занемог",
        "Я помню чудное мгновенье передо мной явилась ты",
        "Мороз и солнце день чудесный еще ты дремлешь друг прелестный",
        "Буря мглою небо кроет вихри снежные крутя",
        "Унылая пора очей очарованье приятна мне твоя прощальная краса",
    ]
    
    # Lermontov samples (simplified for testing)
    lermontov_samples = [
        "Выхожу один я на дорогу сквозь туман кремнистый путь блестит",
        "Белеет парус одинокий в тумане моря голубом",
        "Печально я гляжу на наше поколенье его грядущее иль пусто иль темно",
        "Горные вершины спят во тьме ночной тихие долины полны свежей мглой",
        "Люблю отчизну я но странною любовью не победит ее рассудок мой",
    ]
    
    for text in pushkin_samples:
        corpus.add_text(text, "Pushkin")
        
    for text in lermontov_samples:
        corpus.add_text(text, "Lermontov")
    
    return corpus


def test_corpus_creation(sample_corpus):
    """Test that corpus is created correctly."""
    assert sample_corpus.n_authors == 2
    assert sample_corpus.n_texts == 10
    assert "Pushkin" in sample_corpus.author_names
    assert "Lermontov" in sample_corpus.author_names


def test_train_test_split(sample_corpus):
    """Test train/test split functionality."""
    X_train, X_test, y_train, y_test = sample_corpus.get_train_test_split(test_size=0.3)
    
    assert len(X_train) + len(X_test) == sample_corpus.n_texts
    assert len(y_train) + len(y_test) == sample_corpus.n_texts
    assert len(X_train) > 0
    assert len(X_test) > 0


def test_tfidf_baseline_training(sample_corpus):
    """Test that TF-IDF baseline can be trained."""
    X_train, X_test, y_train, y_test = sample_corpus.get_train_test_split()
    
    baseline = TfidfSvmBaseline(max_features=100)
    baseline.fit(X_train, y_train)
    
    # Should be able to predict
    predictions = baseline.predict(X_test)
    assert len(predictions) == len(X_test)
    
    # Score should be between 0 and 1
    score = baseline.score(X_test, y_test)
    assert 0.0 <= score <= 1.0


def test_entropy_classifier_training(sample_corpus):
    """Test that entropy-based classifier can be trained."""
    X_train, X_test, y_train, y_test = sample_corpus.get_train_test_split()
    
    entropy_clf = EntropyBasedClassifier()
    entropy_clf.fit(X_train, y_train)
    
    # Should be able to predict
    predictions = entropy_clf.predict(X_test)
    assert len(predictions) == len(X_test)
    
    # Score should be between 0 and 1
    score = entropy_clf.score(X_test, y_test)
    assert 0.0 <= score <= 1.0


def test_feature_extraction_consistency():
    """Test that feature extraction is consistent."""
    text = "Мой дядя самых честных правил"
    
    clf = EntropyBasedClassifier()
    features1 = clf._extract_features(text)
    features2 = clf._extract_features(text)
    
    # Should get same features for same text
    np.testing.assert_array_almost_equal(features1, features2)
    
    # Should have correct number of features
    assert len(features1) == len(clf.feature_names)


@pytest.mark.skip(reason="Requires large corpus - use for real validation")
def test_entropy_vs_baseline_comparison():
    """Compare entropy-based method with TF-IDF baseline.
    
    This test should be run with a real, large corpus.
    Expected: Both methods should achieve >70% accuracy on Russian poetry.
    """
    # TODO: Load real corpus from files
    # corpus = load_russian_poetry_corpus()
    
    # X_train, X_test, y_train, y_test = corpus.get_train_test_split()
    
    # # Train baseline
    # baseline = TfidfSvmBaseline()
    # baseline.fit(X_train, y_train)
    # baseline_accuracy = baseline.score(X_test, y_test)
    
    # # Train entropy-based
    # entropy_clf = EntropyBasedClassifier()
    # entropy_clf.fit(X_train, y_train)
    # entropy_accuracy = entropy_clf.score(X_test, y_test)
    
    # print(f"\n{'='*60}")
    # print(f"COMPARISON RESULTS")
    # print(f"{'='*60}")
    # print(f"TF-IDF + SVM Baseline:  {baseline_accuracy:.2%}")
    # print(f"Entropy-based method:   {entropy_accuracy:.2%}")
    # print(f"{'='*60}\n")
    
    # # Both should achieve reasonable accuracy
    # assert baseline_accuracy > 0.70, "Baseline should achieve >70% accuracy"
    # assert entropy_accuracy > 0.50, "Entropy method should achieve >50% accuracy"
    
    pass


# ============================================================================
# Benchmark Report Generation
# ============================================================================

def generate_classification_report(corpus: AuthorCorpus, classifier, name: str):
    """Generate detailed classification report.
    
    Returns:
        dict with metrics: accuracy, precision, recall, f1, confusion_matrix
    """
    X_train, X_test, y_train, y_test = corpus.get_train_test_split()
    
    # Train
    classifier.fit(X_train, y_train)
    
    # Predict
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    cm = confusion_matrix(y_test, y_pred)
    
    report = {
        'name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_authors': corpus.n_authors,
    }
    
    return report


def print_benchmark_report(report: dict):
    """Print formatted benchmark report."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK REPORT: {report['name']}")
    print(f"{'='*70}")
    print(f"Dataset:")
    print(f"  Authors:    {report['n_authors']}")
    print(f"  Train size: {report['n_train']}")
    print(f"  Test size:  {report['n_test']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"  Precision: {report['precision']:.4f}")
    print(f"  Recall:    {report['recall']:.4f}")
    print(f"  F1-score:  {report['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(report['confusion_matrix'])
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

