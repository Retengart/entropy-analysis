use std::fs;
use std::io::{self, Write};
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::model::{Normalization, Summary, SummaryJson};

pub fn print_console_summary(summary: &Summary) -> Result<()> {
    let mut stdout = io::stdout();
    writeln!(stdout, "N = {}", summary.n_words)?;
    writeln!(stdout, "H = {}", optfmt(summary.h_bits))?;
    writeln!(stdout, "x_mean = {}", optfmt(summary.x_mean))?;
    writeln!(stdout, "sigma = {}", optfmt(summary.sigma))?;
    Ok(())
}

fn optfmt(v: Option<f64>) -> String {
    v.map(|x| format!("{:.6}", x)).unwrap_or_else(|| "null".to_string())
}

pub fn write_csv(path: &Path, summary: &Summary) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["rank", "letter", "count", "p_i", "p_i_log2"])?;
    for r in &summary.rows {
        wtr.write_record(&[
            r.rank.to_string(),
            r.letter.to_string(),
            r.count.to_string(),
            format!("{:.10}", r.p),
            format!("{:.10}", r.p_log2),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn write_json(path: &Path, summary: &Summary, norm: &Normalization) -> Result<()> {
    let alphabet_str: String = norm.alphabet.iter().collect();
    let json = SummaryJson {
        n_words: summary.n_words,
        h_bits: summary.h_bits,
        x_mean: summary.x_mean,
        sigma: summary.sigma,
        non_zero_letters: summary.non_zero_letters,
        alphabet: alphabet_str,
        rows: summary.rows.clone(),
        extended_stats: summary.extended_stats.clone(),
        normalized_entropy: summary.normalized_entropy.clone(),
    };
    let s = serde_json::to_string_pretty(&json)?;
    fs::write(path, s)?;
    Ok(())
}

#[derive(Serialize)]
struct SummaryRow<'a> {
    file: &'a str,
    n_words: u64,
    h_bits: Option<f64>,
    x_mean: Option<f64>,
    sigma: Option<f64>,
}

pub fn write_summary_csv(path: &Path, rows: &[(String, Summary)]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["file", "N", "H_bits", "x_mean", "sigma"])?;
    for (file, s) in rows {
        wtr.write_record([
            file.clone(),
            s.n_words.to_string(),
            s.h_bits.map(|v| format!("{:.6}", v)).unwrap_or_default(),
            s.x_mean.map(|v| format!("{:.6}", v)).unwrap_or_default(),
            s.sigma.map(|v| format!("{:.6}", v)).unwrap_or_default(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn write_summary_json(path: &Path, rows: &[(String, Summary)]) -> Result<()> {
    let ser: Vec<SummaryRow> = rows
        .iter()
        .map(|(file, s)| SummaryRow {
            file,
            n_words: s.n_words,
            h_bits: s.h_bits,
            x_mean: s.x_mean,
            sigma: s.sigma,
        })
        .collect();
    let s = serde_json::to_string_pretty(&ser)?;
    fs::write(path, s)?;
    Ok(())
}


