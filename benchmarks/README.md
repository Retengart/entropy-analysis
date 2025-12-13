# Scientific Validation Benchmarks

This directory contains tools for rigorous scientific validation of the entropy-based author attribution method.

## Overview

The benchmark suite compares the entropy-based method against established baseline methods:

- **TF-IDF + SVM** — Standard baseline for authorship attribution
- **Entropy-based** — This project's method using information-theoretic metrics

## Quick Start

### 1. Create a Benchmark Dataset

```bash
# From existing text files (with *** delimiters)
python benchmarks/create_dataset.py \
  --from-files science/pushkin_filtered.txt science/lermontov_filtered.txt \
  --min-words 20 \
  --max-segments 100 \
  --output benchmarks/datasets/pushkin_vs_lermontov.json
```

### 2. Run the Benchmark

```bash
python benchmarks/run_benchmark.py \
  --dataset benchmarks/datasets/pushkin_vs_lermontov.json \
  --output benchmarks/results/report_$(date +%Y%m%d).md
```

### 3. View Results

```bash
cat benchmarks/results/report_*.md
```

## What Gets Measured

### Metrics

- **Accuracy** — Percentage of correctly classified texts
- **Precision** — True positives / (True positives + False positives)
- **Recall** — True positives / (True positives + False negatives)
- **F1-score** — Harmonic mean of precision and recall

### Methods Compared

#### 1. TF-IDF + SVM Baseline

- Character n-grams (1-3)
- TF-IDF vectorization
- Linear SVM classifier
- **Status:** Industry standard, widely used

#### 2. Entropy-based Method

- Features:
  - Shannon entropy
  - Perplexity (2^H)
  - Evenness (Pielou's J)
  - Alphabet utilization
  - Rényi spectrum (H₀, H₂, H∞)
  - Uniqueness ratio
- Classifier: Random Forest
- **Status:** Novel approach being validated

## Dataset Requirements

For reliable results, your dataset should have:

- ✅ **At least 2 authors** (more is better)
- ✅ **At least 50 texts per author** (balanced)
- ✅ **At least 20 words per text** (meaningful statistics)
- ✅ **Similar text types** (all poetry, all prose, etc.)

## Example Output

```
============================================================
SUMMARY
============================================================
TF-IDF + SVM:     0.8523 (85.23%)
Entropy-based:    0.7841 (78.41%)
Difference:       -0.0682 (-6.82%)
============================================================
```

## Interpreting Results

### If Entropy ≥ 80% accuracy:
✅ **Strong performance** — Method is effective for author attribution

### If Entropy 70-80% accuracy:
⚪ **Moderate performance** — Useful for exploratory analysis, consider combining with other methods

### If Entropy < 70% accuracy:
❌ **Limited performance** — Better suited as supplementary features

### Comparison with Baseline:

- **Entropy > Baseline + 5%:** Entropy method is significantly better
- **±5% difference:** Methods are comparable  
- **Entropy < Baseline - 5%:** Baseline is significantly better

## Advanced Usage

### Custom Dataset

```python
from benchmarks.create_dataset import create_dataset_from_files
from pathlib import Path

files = [
    Path('author1_texts.txt'),
    Path('author2_texts.txt'),
    Path('author3_texts.txt'),
]

dataset = create_dataset_from_files(
    files,
    min_words=30,
    max_segments_per_author=100,
)
```

### Programmatic Benchmark

```python
from benchmarks.run_benchmark import run_full_benchmark, load_dataset

corpus, metadata = load_dataset('dataset.json')
results = run_full_benchmark(corpus, metadata)

print(f"Accuracy: {results['methods']['entropy']['accuracy']:.4f}")
```

## Running Tests

```bash
# Run validation tests
pytest tests/test_validation.py -v

# Run specific test
pytest tests/test_validation.py::test_tfidf_baseline_training -v
```

## File Structure

```
benchmarks/
├── README.md                    # This file
├── create_dataset.py            # Dataset creation tool
├── run_benchmark.py             # Benchmark runner
├── datasets/                    # Benchmark datasets (JSON)
│   └── pushkin_vs_lermontov.json
└── results/                     # Benchmark reports (Markdown)
    └── benchmark_report.md
```

## FAQ

### Q: Why is my accuracy low?

**A:** Common reasons:
1. **Dataset too small** — Need 50+ texts per author
2. **Texts too short** — Need 20+ words per text
3. **Authors too similar** — Some authors are genuinely hard to distinguish
4. **Wrong text type** — Method may work better on poetry vs prose

### Q: Should I use this method instead of TF-IDF?

**A:** Not necessarily. Consider:
- **For production:** Use proven methods (TF-IDF, BERT)
- **For research:** Entropy metrics provide interpretable insights
- **Best approach:** Combine both methods as ensemble

### Q: How do I add more authors?

```bash
python benchmarks/create_dataset.py \
  --from-directory my_authors_folder/ \
  --output benchmarks/datasets/multi_author.json
```

Each .txt file in the directory will be treated as one author.

## Contributing

To improve the benchmark suite:

1. Add more baseline methods (e.g., BERT-based)
2. Add cross-validation
3. Add statistical significance tests
4. Add visualization of results

## References

- **Burrows (2002)** — Delta method for authorship attribution
- **Stamatatos (2009)** — Survey of authorship attribution methods
- **Koppel et al. (2014)** — Authorship attribution in the wild

---

*Part of entropy-analysis-py project*

