use std::io::{self, Write};
use std::path::Path;

use anyhow::Result;
use serde::Serialize;
use plotters::coord::Shift;
use plotters::prelude::*;

use crate::model::{Normalization, Summary};
use crate::analyze::analyze_text;

#[derive(Debug, Clone, Serialize)]
pub struct PoemSummary {
    pub title: String,
    pub summary: Summary,
    pub freq_v: f64,  // частота буквы 'в'
    pub freq_n: f64,  // частота буквы 'н'
    pub freq_s: f64,  // частота буквы 'с'
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisTableRow {
    pub num: usize,
    pub title: String,
    pub h_bits: Option<f64>,
    pub zero_letters: usize,
    pub word_count: u64,
    pub freq_v: f64,
    pub freq_n: f64,
    pub freq_s: f64,
}

#[derive(Debug, Clone)]
pub struct TwoAuthorsData {
    pub author1: Vec<PoemSummary>,
    pub author2: Vec<PoemSummary>,
    pub author1_name: String,
    pub author2_name: String,
}

/// Split text by custom delimiter
pub fn split_poems(text: &str, delimiter: &str) -> Vec<(String, String)> {
    let poems: Vec<&str> = text.split(delimiter).collect();
    let mut result = Vec::new();
    
    for (i, poem_text) in poems.iter().enumerate() {
        let trimmed = poem_text.trim();
        if trimmed.is_empty() {
            continue;
        }
        
        // Try to extract title from first line, otherwise use generic title
        let lines: Vec<&str> = trimmed.lines().collect();
        let title = if lines.len() > 1 && lines[0].trim().len() < 100 {
            lines[0].trim().to_string()
        } else {
            format!("Стихотворение {}", i + 1)
        };
        
        result.push((title, trimmed.to_string()));
    }
    
    result
}

/// Analyze multiple poems from a single file
pub fn analyze_multi_poem(text: &str, norm: &Normalization, delimiter: &str) -> Result<Vec<PoemSummary>> {
    let poems = split_poems(text, delimiter);
    let mut results = Vec::new();
    
    for (title, poem_text) in poems {
        let summary = analyze_text(&poem_text, norm)?;
        
        // Find frequencies for specific letters
        let freq_v = find_letter_frequency(&summary, 'в');
        let freq_n = find_letter_frequency(&summary, 'н');
        let freq_s = find_letter_frequency(&summary, 'с');
        
        results.push(PoemSummary {
            title,
            summary,
            freq_v,
            freq_n,
            freq_s,
        });
    }
    
    Ok(results)
}

/// Analyze poems from two authors
pub fn analyze_two_authors(
    text1: &str, 
    text2: &str, 
    norm: &Normalization, 
    delimiter: &str,
    author1_name: &str,
    author2_name: &str
) -> Result<TwoAuthorsData> {
    let author1 = analyze_multi_poem(text1, norm, delimiter)?;
    let author2 = analyze_multi_poem(text2, norm, delimiter)?;
    
    Ok(TwoAuthorsData {
        author1,
        author2,
        author1_name: author1_name.to_string(),
        author2_name: author2_name.to_string(),
    })
}

fn find_letter_frequency(summary: &Summary, letter: char) -> f64 {
    summary.rows.iter()
        .find(|row| row.letter == letter)
        .map(|row| row.p)
        .unwrap_or(0.0)
}

/// Create analysis table in the style of the methodology
pub fn create_analysis_table(poems: &[PoemSummary]) -> Vec<AnalysisTableRow> {
    poems.iter().enumerate().map(|(i, poem)| {
        AnalysisTableRow {
            num: i + 1,
            title: poem.title.clone(),
            h_bits: poem.summary.h_bits,
            zero_letters: poem.summary.rows.iter().filter(|row| row.count == 0).count(),
            word_count: poem.summary.n_words,
            freq_v: poem.freq_v,
            freq_n: poem.freq_n,
            freq_s: poem.freq_s,
        }
    }).collect()
}

/// Write analysis table to CSV
pub fn write_analysis_table_csv(path: &Path, table: &[AnalysisTableRow]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "№",
        "Название стихотворения", 
        "H бит",
        "Число начальных букв с нулевыми частотами",
        "Количество слов в стихотворении",
        "Частоты слов на начальные буквы в",
        "Частоты слов на начальные буквы н", 
        "Частоты слов на начальные буквы с"
    ])?;
    
    for row in table {
        wtr.write_record(&[
            row.num.to_string(),
            row.title.clone(),
            row.h_bits.map(|v| format!("{:.4}", v)).unwrap_or_default(),
            row.zero_letters.to_string(),
            row.word_count.to_string(),
            format!("{:.4}", row.freq_v),
            format!("{:.4}", row.freq_n),
            format!("{:.4}", row.freq_s),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

/// Print analysis table to console
pub fn print_analysis_table(table: &[AnalysisTableRow]) -> Result<()> {
    let mut stdout = io::stdout();
    
    writeln!(stdout, "\nDetailed Analysis Table:")?;
    writeln!(stdout, "{:<3} {:<25} {:<8} {:<10} {:<12} {:<8} {:<8} {:<8}", 
             "№", "Title", "H bits", "Zero", "N words", "freq_в", "freq_н", "freq_с")?;
    writeln!(stdout, "{}", "-".repeat(90))?;
    
    for row in table {
        writeln!(stdout, "{:<3} {:<25} {:<8} {:<10} {:<12} {:<8.4} {:<8.4} {:<8.4}",
                 row.num,
                 if row.title.chars().count() > 25 { 
                     format!("{}...", row.title.chars().take(22).collect::<String>()) 
                 } else { 
                     row.title.clone() 
                 },
                 row.h_bits.map(|v| format!("{:.4}", v)).unwrap_or("-".to_string()),
                 row.zero_letters,
                 row.word_count,
                 row.freq_v,
                 row.freq_n,
                 row.freq_s)?;
    }
    writeln!(stdout)?;
    
    Ok(())
}

/// Plot normal distribution of entropy values
pub fn plot_normal_distribution(path: &Path, poems: &[PoemSummary]) -> Result<()> {
    let entropy_values: Vec<f64> = poems.iter()
        .filter_map(|poem| poem.summary.h_bits)
        .collect();
    
    if entropy_values.len() < 2 {
        return Ok(());
    }
    
    // Calculate statistics
    let n = entropy_values.len() as f64;
    let mean = entropy_values.iter().sum::<f64>() / n;
    let variance = entropy_values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    
    // Determine plot range
    let min_val = entropy_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = entropy_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let padding = (max_val - min_val) * 0.2;
    let x_min = (min_val - padding).max(0.0);
    let x_max = max_val + padding;
    
    // Generate normal distribution curve
    let normal_points: Vec<(f64, f64)> = (0..=200)
        .map(|i| {
            let x = x_min + (x_max - x_min) * (i as f64) / 200.0;
            let y = normal_pdf(x, mean, std_dev);
            (x, y)
        })
        .collect();
    
    let max_density = normal_points.iter().map(|(_, y)| *y).fold(0.0, f64::max);
    
    let (is_svg, path_str) = if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        (ext.eq_ignore_ascii_case("svg"), path.to_string_lossy().to_string())
    } else {
        (false, path.to_string_lossy().to_string())
    };

    if is_svg {
        let root = SVGBackend::new(&path_str, (1200, 800)).into_drawing_area();
        draw_normal_distribution(&root, &entropy_values, &normal_points, mean, std_dev, x_min, x_max, max_density)?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(&path_str, (1200, 800)).into_drawing_area();
        draw_normal_distribution(&root, &entropy_values, &normal_points, mean, std_dev, x_min, x_max, max_density)?;
        root.present()?;
    }
    Ok(())
}

fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let exp_part = -0.5 * ((x - mean) / std_dev).powi(2);
    (1.0 / (std_dev * (2.0 * pi).sqrt())) * exp_part.exp()
}

fn draw_normal_distribution<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    data: &[f64],
    normal_curve: &[(f64, f64)],
    mean: f64,
    std_dev: f64,
    x_min: f64,
    x_max: f64,
    max_density: f64,
) -> Result<()> where DB::ErrorType: 'static {
    root.fill(&WHITE)?;
    
    let title = format!("Normal Distribution of Entropy Values (μ = {:.3}, σ = {:.3})", mean, std_dev);
    
    let mut chart = ChartBuilder::on(root)
        .caption(&title, ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(x_min..x_max, 0.0..(max_density * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Энтропия H (биты)")
        .y_desc("Плотность вероятности")
        .x_label_formatter(&|v| format!("{:.2}", v))
        .y_label_formatter(&|v| format!("{:.3}", v))
        .draw()?;

    // Draw normal distribution curve
    chart.draw_series(LineSeries::new(
        normal_curve.iter().cloned(),
        RED.stroke_width(3),
    ))?.label("Нормальное распределение").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));

    // Draw histogram of actual data
    let bin_count = (data.len() as f64).sqrt().ceil() as usize;
    let bin_width = (x_max - x_min) / bin_count as f64;
    let mut bins = vec![0; bin_count];
    
    for &value in data {
        let bin_index = ((value - x_min) / bin_width).floor() as usize;
        if bin_index < bin_count {
            bins[bin_index] += 1;
        }
    }
    
    // Normalize histogram to density
    let total_area = bin_width * data.len() as f64;
    let histogram_data: Vec<(f64, f64)> = bins.iter().enumerate()
        .map(|(i, &count)| {
            let x = x_min + (i as f64 + 0.5) * bin_width;
            let density = count as f64 / total_area;
            (x, density)
        })
        .collect();

    // Draw histogram bars
    chart.draw_series(
        histogram_data.iter().enumerate().map(|(i, &(_x, density))| {
            let x_start = x_min + i as f64 * bin_width;
            let x_end = x_start + bin_width;
            Rectangle::new([(x_start, 0.0), (x_end, density)], BLUE.mix(0.3).filled().stroke_width(1))
        })
    )?.label("Фактические данные").legend(|(x, y)| Rectangle::new([(x, y), (x + 10, y + 10)], BLUE.mix(0.3).filled()));

    // Draw mean line
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(mean, 0.0), (mean, max_density)],
        GREEN.stroke_width(2),
    )))?.label(&format!("Среднее μ = {:.3}", mean)).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], GREEN));

    chart.configure_series_labels().draw()?;
    
    Ok(())
}
