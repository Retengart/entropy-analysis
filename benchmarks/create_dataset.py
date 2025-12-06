#!/usr/bin/env python3
"""
Create benchmark dataset for author attribution validation.

This script helps collect and organize texts from multiple authors
into a structured dataset for scientific validation.

Usage:
    python benchmarks/create_dataset.py --help
    python benchmarks/create_dataset.py --from-files science/pushkin.txt science/lermontov.txt
    python benchmarks/create_dataset.py --from-directory sample_texts/
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from entropy_analysis.core.normalize import preprocess_text_for_brackets


def split_text_by_delimiter(text: str, delimiter: str = "***") -> List[str]:
    """Split text into segments by delimiter."""
    segments = text.split(delimiter)
    # Clean up segments
    segments = [s.strip() for s in segments if s.strip()]
    return segments


def create_dataset_from_files(
    file_paths: List[Path],
    min_words: int = 20,
    max_segments_per_author: int = None,
) -> Dict:
    """Create benchmark dataset from text files.
    
    Args:
        file_paths: List of paths to text files. Filename should be author name.
        min_words: Minimum number of words per segment (filter out short texts)
        max_segments_per_author: Limit number of segments per author (for balance)
        
    Returns:
        dict with structure:
        {
            'metadata': {...},
            'authors': [name1, name2, ...],
            'texts': [(text, author_id, author_name), ...]
        }
    """
    dataset = {
        'metadata': {
            'min_words': min_words,
            'max_segments_per_author': max_segments_per_author,
        },
        'authors': [],
        'texts': [],
    }
    
    for file_path in file_paths:
        # Extract author name from filename
        author_name = file_path.stem.replace('_filtered', '').replace('_', ' ').title()
        
        print(f"\nProcessing {author_name}...")
        
        # Read and preprocess text
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Preprocess brackets
        content = preprocess_text_for_brackets(content)
        
        # Split into segments
        segments = split_text_by_delimiter(content, delimiter="***")
        
        print(f"  Found {len(segments)} segments")
        
        # Filter segments by word count
        valid_segments = []
        for seg in segments:
            word_count = len(seg.split())
            if word_count >= min_words:
                valid_segments.append(seg)
        
        print(f"  {len(valid_segments)} segments with ≥{min_words} words")
        
        # Limit segments if requested
        if max_segments_per_author and len(valid_segments) > max_segments_per_author:
            valid_segments = valid_segments[:max_segments_per_author]
            print(f"  Limited to {max_segments_per_author} segments")
        
        # Add to dataset
        if author_name not in dataset['authors']:
            dataset['authors'].append(author_name)
        
        author_id = dataset['authors'].index(author_name)
        
        for segment in valid_segments:
            dataset['texts'].append({
                'text': segment,
                'author_id': author_id,
                'author_name': author_name,
                'source_file': str(file_path),
            })
    
    # Add statistics to metadata
    dataset['metadata']['n_authors'] = len(dataset['authors'])
    dataset['metadata']['n_texts'] = len(dataset['texts'])
    dataset['metadata']['texts_per_author'] = {
        author: sum(1 for t in dataset['texts'] if t['author_name'] == author)
        for author in dataset['authors']
    }
    
    return dataset


def save_dataset(dataset: Dict, output_path: Path):
    """Save dataset to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset saved to: {output_path}")


def print_dataset_stats(dataset: Dict):
    """Print dataset statistics."""
    print("\n" + "="*70)
    print("BENCHMARK DATASET STATISTICS")
    print("="*70)
    print(f"Authors: {dataset['metadata']['n_authors']}")
    print(f"Total texts: {dataset['metadata']['n_texts']}")
    print(f"\nTexts per author:")
    for author, count in dataset['metadata']['texts_per_author'].items():
        print(f"  {author:20s}: {count:3d} texts")
    
    # Check if balanced
    counts = list(dataset['metadata']['texts_per_author'].values())
    min_count = min(counts)
    max_count = max(counts)
    balance_ratio = min_count / max_count if max_count > 0 else 0
    
    print(f"\nBalance ratio: {balance_ratio:.2f}")
    if balance_ratio < 0.8:
        print("  ⚠️  WARNING: Dataset is imbalanced! Consider limiting max_segments.")
    else:
        print("  ✅ Dataset is well-balanced")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create benchmark dataset for author attribution validation"
    )
    parser.add_argument(
        '--from-files',
        nargs='+',
        type=Path,
        help='List of text files (one per author)',
    )
    parser.add_argument(
        '--from-directory',
        type=Path,
        help='Directory containing text files',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmarks/datasets/russian_poetry.json'),
        help='Output JSON file path',
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=20,
        help='Minimum words per text segment (default: 20)',
    )
    parser.add_argument(
        '--max-segments',
        type=int,
        default=None,
        help='Maximum segments per author (for balancing)',
    )
    
    args = parser.parse_args()
    
    # Collect file paths
    file_paths = []
    
    if args.from_files:
        file_paths.extend(args.from_files)
    
    if args.from_directory:
        if not args.from_directory.exists():
            print(f"❌ Directory not found: {args.from_directory}")
            return
        file_paths.extend(args.from_directory.glob('*.txt'))
    
    if not file_paths:
        print("❌ No input files specified. Use --from-files or --from-directory")
        parser.print_help()
        return
    
    # Verify files exist
    valid_paths = []
    for path in file_paths:
        if path.exists():
            valid_paths.append(path)
        else:
            print(f"⚠️  File not found, skipping: {path}")
    
    if not valid_paths:
        print("❌ No valid input files found")
        return
    
    print(f"\n📚 Creating dataset from {len(valid_paths)} files...")
    
    # Create dataset
    dataset = create_dataset_from_files(
        valid_paths,
        min_words=args.min_words,
        max_segments_per_author=args.max_segments,
    )
    
    # Print statistics
    print_dataset_stats(dataset)
    
    # Save
    save_dataset(dataset, args.output)
    
    print("✅ Done! You can now use this dataset for validation:")
    print(f"   python benchmarks/run_benchmark.py --dataset {args.output}")


if __name__ == '__main__':
    main()

