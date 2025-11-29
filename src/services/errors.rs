//! Specialized error types for the entropy analysis system
//!
//! This module provides structured error handling with proper error types
//! and better error messages for debugging and user experience.

use std::error::Error as StdError;
use std::fmt;
use thiserror::Error;

/// Main error type for the entropy analysis system
#[derive(Debug, Error)]
pub enum AnalysisError {
    /// IO-related errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid alphabet configuration
    #[error("Invalid alphabet: {alphabet}")]
    InvalidAlphabet { alphabet: String },

    /// Alphabet is empty
    #[error("Alphabet cannot be empty")]
    AlphabetEmpty,

    /// Alphabet contains duplicate characters
    #[error("Alphabet contains duplicate characters: {duplicates:?}")]
    AlphabetDuplicates { duplicates: Vec<char> },

    /// Alphabet too large
    #[error("Alphabet too large: {size} characters (max: {max_size})")]
    AlphabetTooLarge { size: usize, max_size: usize },

    /// File too large
    #[error("File too large: {size} bytes (max: {max_size})")]
    FileTooLarge { size: u64, max_size: u64 },

    /// File not found
    #[error("File not found: {path}")]
    FileNotFound { path: String },

    /// Invalid file path
    #[error("Invalid file path: {path}")]
    InvalidPath { path: String },

    /// Empty file
    #[error("File is empty: {path}")]
    EmptyFile { path: String },

    /// Text normalization error
    #[error("Text normalization error: {message}")]
    Normalization { message: String },

    /// Token processing error
    #[error("Token processing error: {token} at position {position}")]
    TokenProcessing { token: String, position: usize },

    /// Insufficient data for analysis
    #[error("Insufficient data: {message}")]
    InsufficientData { message: String },

    /// Correlation analysis error
    #[error("Correlation analysis failed: {message}")]
    CorrelationError { message: String },

    /// Plot generation error
    #[error("Plot generation failed: {message}")]
    PlotError { message: String },

    /// Output generation error
    #[error("Output generation failed: {format} for path: {path}")]
    OutputError { format: String, path: String },

    /// Configuration error
    #[error("Configuration error: {parameter} = {value} - {message}")]
    Configuration {
        parameter: String,
        value: String,
        message: String,
    },

    /// Regex compilation error
    #[error("Regex compilation error: {pattern}")]
    RegexError { pattern: String },

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// CSV processing error
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// Plotting backend error
    #[error("Plotting backend error: {backend}")]
    PlottingBackend { backend: String },

    /// Invalid parameter value
    #[error("Invalid parameter value: {parameter} = {value}, expected: {expected}")]
    InvalidParameter {
        parameter: String,
        value: String,
        expected: String,
    },

    /// Memory allocation error
    #[error("Memory allocation failed: requested {requested} bytes")]
    MemoryError { requested: usize },

    /// Timeout error
    #[error("Operation timed out after {duration} seconds")]
    Timeout { duration: u64 },

    /// Parallel processing error
    #[error("Parallel processing error: {message}")]
    ParallelError { message: String },
}

impl AnalysisError {
    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(parameter: S, value: S, message: S) -> Self {
        Self::Configuration {
            parameter: parameter.into(),
            value: value.into(),
            message: message.into(),
        }
    }

    /// Create a new normalization error
    pub fn normalization<S: Into<String>>(message: S) -> Self {
        Self::Normalization {
            message: message.into(),
        }
    }

    /// Create a new insufficient data error
    pub fn insufficient_data<S: Into<String>>(message: S) -> Self {
        Self::InsufficientData {
            message: message.into(),
        }
    }

    /// Create a new correlation error
    pub fn correlation<S: Into<String>>(message: S) -> Self {
        Self::CorrelationError {
            message: message.into(),
        }
    }

    /// Create a new plot error
    pub fn plot<S: Into<String>>(message: S) -> Self {
        Self::PlotError {
            message: message.into(),
        }
    }

    /// Create a new output error
    pub fn output<S: Into<String>>(format: S, path: S) -> Self {
        Self::OutputError {
            format: format.into(),
            path: path.into(),
        }
    }

    /// Create a new token processing error
    pub fn token_processing<S: Into<String>>(token: S, position: usize) -> Self {
        Self::TokenProcessing {
            token: token.into(),
            position,
        }
    }

    /// Create a new invalid parameter error
    pub fn invalid_parameter<S: Into<String>>(parameter: S, value: S, expected: S) -> Self {
        Self::InvalidParameter {
            parameter: parameter.into(),
            value: value.into(),
            expected: expected.into(),
        }
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::FileNotFound { .. }
                | Self::EmptyFile { .. }
                | Self::InvalidPath { .. }
                | Self::Normalization { .. }
                | Self::InsufficientData { .. }
        )
    }

    /// Check if this is a user error (vs system error)
    pub fn is_user_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidAlphabet { .. }
                | Self::AlphabetEmpty
                | Self::AlphabetDuplicates { .. }
                | Self::AlphabetTooLarge { .. }
                | Self::InvalidPath { .. }
                | Self::Configuration { .. }
                | Self::InvalidParameter { .. }
        )
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::MemoryError { .. } | Self::Timeout { .. } => ErrorSeverity::Critical,
            Self::Io(_) | Self::FileTooLarge { .. } | Self::ParallelError { .. } => {
                ErrorSeverity::High
            }
            Self::InvalidAlphabet { .. }
            | Self::AlphabetEmpty
            | Self::AlphabetDuplicates { .. }
            | Self::AlphabetTooLarge { .. }
            | Self::FileNotFound { .. }
            | Self::InvalidPath { .. }
            | Self::OutputError { .. }
            | Self::PlottingBackend { .. }
            | Self::Configuration { .. }
            | Self::RegexError { .. } => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }

    /// Get suggestions for fixing the error
    pub fn suggestions(&self) -> Vec<String> {
        match self {
            Self::FileNotFound { path } => vec![
                format!("Check if the file path '{}' exists", path),
                "Verify the file permissions".to_string(),
                "Check for typos in the file path".to_string(),
            ],
            Self::FileTooLarge { size, max_size } => vec![
                format!(
                    "Use a smaller file (current: {} bytes, max: {} bytes)",
                    size, max_size
                ),
                "Consider splitting the file into smaller chunks".to_string(),
                "Use streaming mode for large files".to_string(),
            ],
            Self::AlphabetEmpty => vec![
                "Provide a non-empty alphabet".to_string(),
                "Use one of the predefined alphabets: rus28, rus33".to_string(),
            ],
            Self::AlphabetDuplicates { duplicates } => vec![
                format!("Remove duplicate characters: {:?}", duplicates),
                "Check the alphabet definition for repeated characters".to_string(),
            ],
            Self::InsufficientData { message } => vec![
                format!("Provide more data: {}", message),
                "Check if the input file contains valid text".to_string(),
                "Verify the text normalization settings".to_string(),
            ],
            Self::Configuration {
                parameter,
                value,
                message,
                ..
            } => vec![
                format!("Set '{}' to a valid value: {}", parameter, message),
                "Check the documentation for valid parameter ranges".to_string(),
            ],
            _ => vec![
                "Check the error message for details".to_string(),
                "Review the documentation for proper usage".to_string(),
            ],
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Validation utilities for preventing errors
pub struct Validation;

impl Validation {
    /// Validate alphabet configuration
    pub fn validate_alphabet(alphabet: &[char]) -> Result<(), AnalysisError> {
        if alphabet.is_empty() {
            return Err(AnalysisError::AlphabetEmpty);
        }

        if alphabet.len() > 100 {
            return Err(AnalysisError::AlphabetTooLarge {
                size: alphabet.len(),
                max_size: 100,
            });
        }

        let mut seen = std::collections::HashSet::new();
        let mut duplicates = Vec::new();

        for &ch in alphabet {
            if seen.contains(&ch) {
                duplicates.push(ch);
            } else {
                seen.insert(ch);
            }
        }

        if !duplicates.is_empty() {
            return Err(AnalysisError::AlphabetDuplicates { duplicates });
        }

        Ok(())
    }

    /// Validate file size
    pub fn validate_file_size(size: u64, max_size: u64) -> Result<(), AnalysisError> {
        if size > max_size {
            return Err(AnalysisError::FileTooLarge { size, max_size });
        }
        Ok(())
    }

    /// Validate parameter ranges
    pub fn validate_range<T: PartialOrd + fmt::Display>(
        value: T,
        min: T,
        max: T,
        parameter: &str,
    ) -> Result<(), AnalysisError> {
        if value < min || value > max {
            return Err(AnalysisError::InvalidParameter {
                parameter: parameter.to_string(),
                value: value.to_string(),
                expected: format!("{}..{}", min, max),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_alphabet() {
        // Valid alphabet
        assert!(Validation::validate_alphabet(&['а', 'б', 'в']).is_ok());

        // Empty alphabet
        assert!(matches!(
            Validation::validate_alphabet(&[]),
            Err(AnalysisError::AlphabetEmpty)
        ));

        // Duplicate characters
        assert!(matches!(
            Validation::validate_alphabet(&['а', 'б', 'а']),
            Err(AnalysisError::AlphabetDuplicates { .. })
        ));

        // Too large alphabet
        let large_alphabet: Vec<char> = (0..101).map(|_| 'а').collect();
        assert!(matches!(
            Validation::validate_alphabet(&large_alphabet),
            Err(AnalysisError::AlphabetTooLarge { .. })
        ));
    }

    #[test]
    fn test_validate_file_size() {
        assert!(Validation::validate_file_size(1000, 10000).is_ok());
        assert!(matches!(
            Validation::validate_file_size(15000, 10000),
            Err(AnalysisError::FileTooLarge { .. })
        ));
    }

    #[test]
    fn test_validate_range() {
        assert!(Validation::validate_range(5, 1, 10, "test_param").is_ok());
        assert!(matches!(
            Validation::validate_range(15, 1, 10, "test_param"),
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn test_error_severity() {
        let error = AnalysisError::MemoryError { requested: 1000 };
        assert_eq!(error.severity(), ErrorSeverity::Critical);

        let error = AnalysisError::FileNotFound {
            path: "test.txt".to_string(),
        };
        assert_eq!(error.severity(), ErrorSeverity::Medium);
    }

    #[test]
    fn test_error_suggestions() {
        let error = AnalysisError::FileNotFound {
            path: "missing.txt".to_string(),
        };
        let suggestions = error.suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].contains("missing.txt"));
    }

    #[test]
    fn test_error_categories() {
        let user_error = AnalysisError::InvalidAlphabet {
            alphabet: "xyz".to_string(),
        };
        assert!(user_error.is_user_error());
        assert!(!user_error.is_recoverable());

        let recoverable = AnalysisError::FileNotFound {
            path: "test.txt".to_string(),
        };
        assert!(recoverable.is_recoverable());
    }
}
