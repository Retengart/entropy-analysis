//! TextAnalyzer service for modular entropy analysis
//!
//! This module provides a clean separation of concerns for text analysis,
//! moving the core logic out of main.rs and into a reusable service.

use anyhow::{Context, Result};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::analyze::analyze_text;
use crate::cli::AlphabetChoice;
use crate::model::{Normalization, Summary};
use crate::normalize::build_normalization;

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub dir: PathBuf,
    pub recurse: bool,
    pub extension: String,
    pub alphabet: AlphabetChoice,
    pub custom_alphabet: Option<String>,
    pub keep_yo: bool,
    pub keep_j: bool,
    pub parallel: bool,
}

/// Results from batch analysis
#[derive(Debug)]
pub struct BatchResult {
    pub summaries: Vec<(String, Summary)>,
    pub total_files: usize,
    pub processed_files: usize,
    pub failed_files: Vec<(String, String)>, // (path, error)
}

/// Service for analyzing text files
pub struct TextAnalyzer {
    normalization: Normalization,
    parallel: bool,
}

impl TextAnalyzer {
    /// Create a new TextAnalyzer with the given configuration
    pub fn new(
        alphabet: AlphabetChoice,
        custom_alphabet: Option<String>,
        keep_yo: bool,
        keep_j: bool,
        parallel: bool,
    ) -> Result<Self> {
        let normalization = build_normalization(
            alphabet,
            custom_alphabet.as_deref(),
            keep_yo,
            keep_j,
            1, // min_token_len
        )?;

        Ok(Self {
            normalization,
            parallel,
        })
    }

    /// Analyze a single text string
    pub fn analyze_text(&self, text: &str) -> Result<Summary> {
        analyze_text(text, &self.normalization).context("Failed to analyze text")
    }

    /// Analyze text from a reader (streaming for large files)
    pub fn analyze_reader(&self, reader: impl BufRead) -> Result<Summary> {
        let text = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");

        self.analyze_text(&text)
    }

    /// Analyze a single file
    pub fn analyze_file(&self, path: &Path) -> Result<(String, Summary)> {
        let file_path = path.to_string_lossy().to_string();

        // Check file size to avoid memory issues
        let metadata = fs::metadata(path)
            .with_context(|| format!("Cannot read metadata for file: {}", file_path))?;

        if metadata.len() > 100 * 1024 * 1024 {
            // 100MB limit
            anyhow::bail!("File too large: {} bytes (max: 100MB)", metadata.len());
        }

        // For large files, use streaming
        let summary = if metadata.len() > 10 * 1024 * 1024 {
            // 10MB threshold
            let file =
                fs::File::open(path).with_context(|| format!("Cannot open file: {}", file_path))?;
            let reader = BufReader::new(file);
            self.analyze_reader(reader)?
        } else {
            let content = fs::read_to_string(path)
                .with_context(|| format!("Cannot read file: {}", file_path))?;
            self.analyze_text(&content)?
        };

        Ok((file_path, summary))
    }

    /// Process multiple files in batch
    pub fn analyze_batch(&self, config: &BatchConfig) -> Result<BatchResult> {
        let files = self.discover_files(config)?;
        let total_files = files.len();

        if self.parallel && total_files > 1 {
            self.analyze_batch_parallel(files)
        } else {
            self.analyze_batch_sequential(files)
        }
        .map(|(summaries, failed_files)| BatchResult {
            summaries,
            total_files,
            processed_files: total_files - failed_files.len(),
            failed_files,
        })
    }

    /// Discover files matching the criteria
    fn discover_files(&self, config: &BatchConfig) -> Result<Vec<PathBuf>> {
        let max_depth = if config.recurse { usize::MAX } else { 1 };

        let walker = WalkDir::new(&config.dir)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_type().is_file()
                    && entry
                        .path()
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|ext| ext == config.extension)
                        .unwrap_or(false)
            });

        Ok(walker.map(|entry| entry.path().to_path_buf()).collect())
    }

    /// Sequential batch processing
    fn analyze_batch_sequential(
        &self,
        files: Vec<PathBuf>,
    ) -> Result<(Vec<(String, Summary)>, Vec<(String, String)>)> {
        let mut summaries = Vec::new();
        let mut failed_files = Vec::new();

        for file_path in files {
            match self.analyze_file(&file_path) {
                Ok(result) => summaries.push(result),
                Err(e) => {
                    let path_str = file_path.to_string_lossy().to_string();
                    failed_files.push((path_str, e.to_string()));
                }
            }
        }

        Ok((summaries, failed_files))
    }

    /// Parallel batch processing using rayon
    #[cfg(feature = "parallel")]
    fn analyze_batch_parallel(
        &self,
        files: Vec<PathBuf>,
    ) -> Result<(Vec<(String, Summary)>, Vec<(String, String)>)> {
        use rayon::prelude::*;

        let results: Vec<Result<(String, Summary)>> = files
            .par_iter()
            .map(|file_path| self.analyze_file(file_path))
            .collect();

        let mut summaries = Vec::new();
        let mut failed_files = Vec::new();

        for (file_path, result) in files.into_iter().zip(results) {
            match result {
                Ok(summary) => summaries.push(summary),
                Err(e) => {
                    let path_str = file_path.to_string_lossy().to_string();
                    failed_files.push((path_str, e.to_string()));
                }
            }
        }

        Ok((summaries, failed_files))
    }

    /// Fallback for when parallel feature is not enabled
    #[cfg(not(feature = "parallel"))]
    fn analyze_batch_parallel(
        &self,
        files: Vec<PathBuf>,
    ) -> Result<(Vec<(String, Summary)>, Vec<(String, String)>)> {
        // Fall back to sequential processing
        self.analyze_batch_sequential(files)
    }

    /// Get the current normalization configuration
    pub fn normalization(&self) -> &Normalization {
        &self.normalization
    }
}

/// Builder for TextAnalyzer
pub struct TextAnalyzerBuilder {
    alphabet: AlphabetChoice,
    custom_alphabet: Option<String>,
    keep_yo: bool,
    keep_j: bool,
    parallel: bool,
}

impl TextAnalyzerBuilder {
    pub fn new() -> Self {
        Self {
            alphabet: AlphabetChoice::Rus28,
            custom_alphabet: None,
            keep_yo: false,
            keep_j: false,
            parallel: true,
        }
    }

    pub fn alphabet(mut self, alphabet: AlphabetChoice) -> Self {
        self.alphabet = alphabet;
        self
    }

    pub fn custom_alphabet<S: Into<String>>(mut self, alphabet: S) -> Self {
        self.custom_alphabet = Some(alphabet.into());
        self
    }

    pub fn keep_yo(mut self, keep_yo: bool) -> Self {
        self.keep_yo = keep_yo;
        self
    }

    pub fn keep_j(mut self, keep_j: bool) -> Self {
        self.keep_j = keep_j;
        self
    }

    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn build(self) -> Result<TextAnalyzer> {
        TextAnalyzer::new(
            self.alphabet,
            self.custom_alphabet,
            self.keep_yo,
            self.keep_j,
            self.parallel,
        )
    }
}

impl Default for TextAnalyzerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_text_analyzer_creation() {
        let analyzer = TextAnalyzerBuilder::new()
            .alphabet(AlphabetChoice::Rus28)
            .keep_yo(true)
            .build();

        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_analyze_simple_text() {
        let analyzer = TextAnalyzerBuilder::new().build().unwrap();

        let text = "а б в г д";
        let summary = analyzer.analyze_text(text).unwrap();

        assert_eq!(summary.n_words, 5);
        assert!(summary.h_bits.is_some());
        assert!(summary.h_bits.unwrap() > 2.0); // 5 different letters
    }

    #[test]
    fn test_analyze_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        fs::write(&file_path, "а б в г д").unwrap();

        let analyzer = TextAnalyzerBuilder::new().build().unwrap();
        let (path, summary) = analyzer.analyze_file(&file_path).unwrap();

        assert_eq!(summary.n_words, 5);
        assert!(path.contains("test.txt"));
    }

    #[test]
    fn test_discover_files() {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        fs::write(temp_dir.path().join("test1.txt"), "test").unwrap();
        fs::write(temp_dir.path().join("test2.txt"), "test").unwrap();
        fs::write(temp_dir.path().join("test.md"), "markdown").unwrap(); // Different extension

        let config = BatchConfig {
            dir: temp_dir.path().to_path_buf(),
            recurse: false,
            extension: "txt".to_string(),
            alphabet: AlphabetChoice::Rus28,
            custom_alphabet: None,
            keep_yo: false,
            keep_j: false,
            parallel: false,
        };

        let analyzer = TextAnalyzerBuilder::new().build().unwrap();
        let files = analyzer.discover_files(&config).unwrap();

        assert_eq!(files.len(), 2); // Only .txt files
    }
}
