use anyhow::Result;
use regex::Regex;

use crate::model::{
    ExtendedStatsSummary, LetterRow, Normalization, NormalizedEntropySummary, Summary,
};
use crate::services::errors::AnalysisError;
use crate::stats::{calculate_extended_stats, calculate_normalized_entropy};

pub fn analyze_text(text: &str, norm: &Normalization) -> Result<Summary> {
    let letters = &norm.alphabet;
    let mut counts = vec![0u64; letters.len()];
    let mut n_words: u64 = 0;

    let word_re = Regex::new("(?iu)[а-яё]+")?; // Unicode, case-insensitive

    for m in word_re.find_iter(text) {
        let token = m.as_str();
        if token.chars().count() < norm.min_token_len {
            continue;
        }

        // Safe token processing
        let mut ch = token
            .chars()
            .next()
            .and_then(|c| c.to_lowercase().next())
            .ok_or_else(|| AnalysisError::token_processing(token, 0))?;

        // Normalization for rus28-like behavior
        if norm.alphabet.len() == 28 {
            if ch == 'ё' && !norm.rus28_keep_yo {
                ch = 'е';
            }
            if ch == 'й' && !norm.rus28_keep_j {
                ch = 'и';
            }
            if ch == 'ъ' || ch == 'ь' {
                continue;
            }
        }

        if let Some(idx) = letters.iter().position(|&c| c == ch) {
            counts[idx] += 1;
            n_words += 1;
        }
    }

    // Compute probabilities and metrics
    let mut rows = Vec::with_capacity(letters.len());
    let mut h_bits = None;
    let mut x_mean = None;
    let mut sigma = None;
    let mut non_zero = 0usize;

    if n_words > 0 {
        let n_f = n_words as f64;
        let mut h_acc = 0.0f64;
        for (i, (letter, &cnt)) in letters.iter().zip(counts.iter()).enumerate() {
            let p = cnt as f64 / n_f;
            let p_log2 = if p > 0.0 { p * p.log2() } else { 0.0 };
            if p > 0.0 {
                non_zero += 1;
            }
            h_acc += p_log2;
            rows.push(LetterRow {
                rank: i + 1,
                letter: *letter,
                count: cnt,
                p,
                p_log2,
            });
        }
        let h = -h_acc;
        h_bits = Some(h);

        // mean and sigma in rank units
        let mut mean = 0.0;
        for r in &rows {
            mean += (r.rank as f64) * r.p;
        }
        let mut var = 0.0;
        for r in &rows {
            let d = (r.rank as f64) - mean;
            var += d * d * r.p;
        }
        x_mean = Some(mean);
        sigma = Some(var.sqrt());
    } else {
        for (i, letter) in letters.iter().enumerate() {
            rows.push(LetterRow {
                rank: i + 1,
                letter: *letter,
                count: 0,
                p: 0.0,
                p_log2: 0.0,
            });
        }
    }

    Ok(Summary {
        n_words,
        h_bits,
        x_mean,
        sigma,
        non_zero_letters: non_zero,
        rows,
        extended_stats: None,
        normalized_entropy: None,
    })
}

/// Enrich summary with normalized entropy metrics
pub fn add_normalized_entropy(summary: &mut Summary) {
    if let Some(h) = summary.h_bits {
        let norm = calculate_normalized_entropy(h, summary.n_words);
        summary.normalized_entropy = Some(NormalizedEntropySummary {
            h_normalized: Some(norm.h_norm),
            entropy_rate: Some(norm.entropy_rate),
        });
    }
}

/// Calculate extended statistics for a batch of summaries
pub fn calculate_batch_extended_stats(
    summaries: &[(String, Summary)],
) -> Option<ExtendedStatsSummary> {
    // Collect all entropy values
    let entropy_values: Vec<f64> = summaries.iter().filter_map(|(_, s)| s.h_bits).collect();

    if entropy_values.len() < 2 {
        return None;
    }

    let stats = calculate_extended_stats(&entropy_values)?;

    Some(ExtendedStatsSummary {
        median: Some(stats.median),
        q1: Some(stats.q1),
        q3: Some(stats.q3),
        iqr: Some(stats.iqr),
        variance: Some(stats.variance),
        std_dev: Some(stats.std_dev),
        coefficient_of_variation: Some(stats.coefficient_of_variation),
        skewness: Some(stats.skewness),
        kurtosis: Some(stats.kurtosis),
    })
}
