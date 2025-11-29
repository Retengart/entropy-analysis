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

#[derive(Debug, Clone)]
pub struct TranslationSystem {
    pub name: String,
    pub poems: Vec<PoemSummary>,
}

#[derive(Debug, Clone)]
pub struct TranslationCorrelation {
    pub system1: String,
    pub system2: String,
    pub correlation: f64,
    pub n_samples: usize,
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

/// Split text by custom delimiter and identify translation systems
pub fn split_poems_by_translation_system(text: &str, delimiter: &str) -> Vec<(String, String, String)> {
    let poems: Vec<&str> = text.split(delimiter).collect();
    let mut result = Vec::new();

    for (i, poem_text) in poems.iter().enumerate() {
        let trimmed = poem_text.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Try to extract translation system and title from first line
        let lines: Vec<&str> = trimmed.lines().collect();
        let (system, title) = if lines.len() > 1 && lines[0].trim().len() < 100 {
            let first_line = lines[0].trim();
            // Look for patterns like "Яндекс - Title" or "Google - Title"
            if let Some((system_part, title_part)) = first_line.split_once(" - ") {
                (system_part.trim().to_string(), title_part.trim().to_string())
            } else if let Some((system_part, title_part)) = first_line.split_once(" — ") {
                (system_part.trim().to_string(), title_part.trim().to_string())
            } else {
                // No clear system-title separator, use generic
                ("Неизвестная система".to_string(), first_line.to_string())
            }
        } else {
            (format!("Система {}", i + 1), format!("Стихотворение {}", i + 1))
        };

        result.push((system, title, trimmed.to_string()));
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

/// Analyze poems by translation systems and calculate correlations between them
pub fn analyze_translation_systems(
    text: &str,
    norm: &Normalization,
    delimiter: &str,
) -> Result<Vec<TranslationSystem>> {
    let poem_blocks = split_poems_by_translation_system(text, delimiter);
    let mut system_map: std::collections::HashMap<String, Vec<(String, String)>> = std::collections::HashMap::new();

    // Group poems by translation system
    for (system, title, poem_text) in poem_blocks {
        system_map.entry(system).or_insert_with(Vec::new).push((title, poem_text));
    }

    let mut results = Vec::new();

    for (system_name, poems) in system_map {
        let mut poem_summaries = Vec::new();

        for (title, poem_text) in poems {
            // Skip the first line if it contains system name
            let text_to_analyze = if poem_text.lines().count() > 1 {
                poem_text.lines().skip(1).collect::<Vec<_>>().join("\n")
            } else {
                poem_text
            };

            if let Ok(summary) = analyze_text(&text_to_analyze, norm) {
                let freq_v = find_letter_frequency(&summary, 'в');
                let freq_n = find_letter_frequency(&summary, 'н');
                let freq_s = find_letter_frequency(&summary, 'с');

                poem_summaries.push(PoemSummary {
                    title,
                    summary,
                    freq_v,
                    freq_n,
                    freq_s,
                });
            }
        }

        if !poem_summaries.is_empty() {
            results.push(TranslationSystem {
                name: system_name,
                poems: poem_summaries,
            });
        }
    }

    Ok(results)
}

/// Calculate correlation between two translation systems
pub fn calculate_translation_correlation(
    system1: &TranslationSystem,
    system2: &TranslationSystem,
) -> Option<TranslationCorrelation> {
    // Create maps of poem titles to entropy values
    let mut entropy1: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut entropy2: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for poem in &system1.poems {
        if let Some(h) = poem.summary.h_bits {
            entropy1.insert(poem.title.clone(), h);
        }
    }

    for poem in &system2.poems {
        if let Some(h) = poem.summary.h_bits {
            entropy2.insert(poem.title.clone(), h);
        }
    }

    // Find common poems
    let common_poems: Vec<String> = entropy1.keys()
        .filter(|title| entropy2.contains_key(*title))
        .cloned()
        .collect();

    if common_poems.len() < 2 {
        return None;
    }

    let values1: Vec<f64> = common_poems.iter()
        .map(|title| entropy1[title])
        .collect();
    let values2: Vec<f64> = common_poems.iter()
        .map(|title| entropy2[title])
        .collect();

    // Calculate Pearson correlation
    let n = values1.len() as f64;
    let mean1 = values1.iter().sum::<f64>() / n;
    let mean2 = values2.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;

    for (v1, v2) in values1.iter().zip(values2.iter()) {
        let diff1 = v1 - mean1;
        let diff2 = v2 - mean2;
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }

    if sum_sq1 == 0.0 || sum_sq2 == 0.0 {
        return None;
    }

    let correlation = numerator / (sum_sq1 * sum_sq2).sqrt();

    Some(TranslationCorrelation {
        system1: system1.name.clone(),
        system2: system2.name.clone(),
        correlation,
        n_samples: common_poems.len(),
    })
}

/// Plot correlation between translation systems
pub fn plot_correlation_lang(path: &Path, poems: &[PoemSummary]) -> Result<()> {
    // Group poems by translation system
    let mut system_groups: std::collections::HashMap<String, Vec<(String, f64)>> = std::collections::HashMap::new();

    for poem in poems {
        // Determine which translation system this poem belongs to
        let mut system_name = "Оригинал".to_string();

        // Check if title contains system indicators
        if poem.title.contains("Яндекс") || poem.title.contains("Yandex") {
            system_name = "Яндекс".to_string();
        } else if poem.title.contains("Google") || poem.title.contains("Гугл") {
            system_name = "Google".to_string();
        } else if poem.title.contains("Deepl") || poem.title.contains("DeepL") {
            system_name = "DeepL".to_string();
        } else if poem.title.contains("Microsoft") || poem.title.contains("Bing") {
            system_name = "Microsoft".to_string();
        } else if poem.title.contains("jp") || poem.title.contains("японск") {
            system_name = "Японский перевод".to_string();
        }

        if let Some(h) = poem.summary.h_bits {
            // Clean the title by removing system prefix for better matching
            let clean_title = if system_name != "Оригинал" {
                poem.title.replace(&format!("{} - ", system_name), "")
                         .replace(&format!("{} — ", system_name), "")
                         .trim().to_string()
            } else {
                poem.title.replace("Оригинал - ", "").trim().to_string()
            };
            system_groups.entry(system_name).or_insert_with(Vec::new).push((clean_title, h));
        }
    }

    // If we don't have multiple systems, fall back to simple entropy distribution
    if system_groups.len() < 2 {
        let entropy_values: Vec<f64> = poems.iter()
            .filter_map(|poem| poem.summary.h_bits)
            .collect();

        if entropy_values.len() < 2 {
            return Ok(());
        }

        let n = entropy_values.len() as f64;
        let mean = entropy_values.iter().sum::<f64>() / n;
        let variance = entropy_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let min_val = entropy_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = entropy_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let (is_svg, path_str) = if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            (ext.eq_ignore_ascii_case("svg"), path.to_string_lossy().to_string())
        } else {
            (false, path.to_string_lossy().to_string())
        };

        if is_svg {
            let root = SVGBackend::new(&path_str, (1200, 800)).into_drawing_area();
            draw_correlation_lang_plot(&root, &entropy_values, mean, std_dev, min_val, max_val)?;
            root.present()?;
        } else {
            let root = BitMapBackend::new(&path_str, (1200, 800)).into_drawing_area();
            draw_correlation_lang_plot(&root, &entropy_values, mean, std_dev, min_val, max_val)?;
            root.present()?;
        }
        return Ok(());
    }

    // Calculate correlations between original and each translation system
    let original_poems = system_groups.get("Оригинал").cloned().unwrap_or_default();

    if original_poems.is_empty() {
        return Ok(());
    }

    // Create a map of poem titles to entropy values for originals
    let mut original_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (title, entropy) in original_poems {
        original_map.insert(title, entropy);
    }

    // Calculate correlation for each translation system
    let mut system_correlations = Vec::new();

    // Add original correlation (self-correlation = 1.0) at position 0
    system_correlations.push(("Оригинал".to_string(), 1.0));

    for (system_name, translated_poems) in &system_groups {
        if system_name == "Оригинал" {
            continue;
        }

        // Create map for translated poems
        let mut translated_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for (title, entropy) in translated_poems {
            translated_map.insert(title.clone(), *entropy);
        }

        // Find matching poems and calculate correlation
        let mut original_values = Vec::new();
        let mut translated_values = Vec::new();

        for (orig_title, orig_entropy) in &original_map {
            if let Some(trans_entropy) = translated_map.get(orig_title) {
                original_values.push(*orig_entropy);
                translated_values.push(*trans_entropy);
            }
        }

        if original_values.len() >= 2 {
            if let Some(correlation) = calculate_correlation_between_systems(&original_values, &translated_values) {
                system_correlations.push((system_name.clone(), correlation));
            }
        }
    }

    if system_correlations.len() < 2 {
        return Ok(());
    }

    // Plot correlation results as a line/bar chart
    plot_correlation_line_chart(path, &system_correlations)?;

    Ok(())
}

/// Calculate correlation between original poems and translated poems
fn calculate_translation_correlation_with_original(
    original_poems: &[(String, f64)],
    translated_poems: &[(String, f64)],
) -> Option<f64> {
    if original_poems.is_empty() || translated_poems.is_empty() {
        return None;
    }

    // Create maps for easier lookup
    let mut original_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (title, entropy) in original_poems {
        original_map.insert(title.clone(), *entropy);
    }

    let mut translated_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (title, entropy) in translated_poems {
        translated_map.insert(title.clone(), *entropy);
    }

    // Find common poems (by title similarity)
    let mut common_values: Vec<(f64, f64)> = Vec::new();

    for (orig_title, orig_entropy) in &original_map {
        // Look for translated version of the same poem
        for (trans_title, trans_entropy) in &translated_map {
            if titles_are_similar(orig_title, trans_title) {
                common_values.push((*orig_entropy, *trans_entropy));
                break;
            }
        }
    }

    if common_values.len() < 2 {
        return None;
    }

    // Calculate correlation between original and translated entropies
    calculate_correlation_between_systems(
        &common_values.iter().map(|(orig, _)| *orig).collect::<Vec<_>>(),
        &common_values.iter().map(|(_, trans)| *trans).collect::<Vec<_>>()
    )
}

/// Check if two titles refer to the same poem
fn titles_are_similar(title1: &str, title2: &str) -> bool {
    // Simple heuristic: check if both titles contain the same key words
    let title1_lower = title1.to_lowercase();
    let title2_lower = title2.to_lowercase();

    let words1: std::collections::HashSet<String> = title1_lower.split_whitespace()
        .filter(|w| w.len() > 3) // Only meaningful words
        .map(|s| s.to_string())
        .collect();
    let words2: std::collections::HashSet<String> = title2_lower.split_whitespace()
        .filter(|w| w.len() > 3)
        .map(|s| s.to_string())
        .collect();

    // Calculate Jaccard similarity
    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    if union == 0 {
        return false;
    }

    let similarity = intersection as f64 / union as f64;
    similarity > 0.3 // Threshold for similarity
}

/// Plot correlation results as a line chart
fn plot_correlation_line_chart(path: &Path, correlations: &[(String, f64)]) -> Result<()> {
    let (is_svg, path_str) = if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        (ext.eq_ignore_ascii_case("svg"), path.to_string_lossy().to_string())
    } else {
        (false, path.to_string_lossy().to_string())
    };

    // Prepare data for plotting - sort by correlation value
    let mut plot_data: Vec<(String, f64)> = correlations.iter()
        .map(|(system, corr)| (system.clone(), *corr))
        .collect();

    plot_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_corr = plot_data.iter().map(|(_, corr)| corr.abs()).fold(0.0, f64::max);
    let y_max = if max_corr > 0.0 { max_corr * 1.1 } else { 1.0 };
    let y_min = if max_corr > 0.0 { -0.1 } else { -1.0 };

    if is_svg {
        let root = SVGBackend::new(&path_str, (1200, 800)).into_drawing_area();
        draw_correlation_line(&root, &plot_data, y_min, y_max)?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(&path_str, (1200, 800)).into_drawing_area();
        draw_correlation_line(&root, &plot_data, y_min, y_max)?;
        root.present()?;
    }
    Ok(())
}

/// Calculate correlation between two sets of entropy values
fn calculate_correlation_between_systems(values1: &[f64], values2: &[f64]) -> Option<f64> {
    if values1.len() != values2.len() || values1.len() < 2 {
        return None;
    }

    let n = values1.len() as f64;
    let mean1 = values1.iter().sum::<f64>() / n;
    let mean2 = values2.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;

    for (v1, v2) in values1.iter().zip(values2.iter()) {
        let diff1 = v1 - mean1;
        let diff2 = v2 - mean2;
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }

    if sum_sq1 == 0.0 || sum_sq2 == 0.0 {
        return None;
    }

    Some(numerator / (sum_sq1 * sum_sq2).sqrt())
}


fn draw_correlation_line<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    data: &[(String, f64)],
    y_min: f64,
    y_max: f64,
) -> Result<()> where DB::ErrorType: 'static {
    root.fill(&WHITE)?;

    let title = "Корреляция между системами перевода".to_string();

    let mut chart = ChartBuilder::on(root)
        .caption(&title, ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(100)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..data.len() as f64, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Системы перевода")
        .y_desc("Коэффициент корреляции")
        .x_label_formatter(&|x| {
            let i = *x as usize;
            if i < data.len() {
                data[i].0.clone()
            } else {
                x.to_string()
            }
        })
        .y_label_formatter(&|v| format!("{:.3}", v))
        .draw()?;

    // Draw zero line
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (data.len() as f64, 0.0)],
        BLACK.stroke_width(2),
    )))?;

    // Draw data points and line
    let points: Vec<(f64, f64)> = data.iter()
        .enumerate()
        .map(|(i, (_, corr))| (i as f64, *corr))
        .collect();

    // Draw line connecting points
    if points.len() > 1 {
        chart.draw_series(LineSeries::new(
            points.iter().cloned(),
            BLUE.stroke_width(3),
        ))?;
    }

    // Draw data points
    for (i, (_, corr)) in data.iter().enumerate() {
        let color = if *corr > 0.0 { RGBColor(0, 150, 0) } else { RGBColor(150, 0, 0) };
        chart.draw_series(std::iter::once(
            Circle::new((i as f64, *corr), 8, color.filled())
        ))?;

        // Add value labels above points
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.3}", corr),
            (i as f64, *corr + 0.05),
            ("sans-serif", 20).into_font().color(&BLACK),
        )))?;
    }

    Ok(())
}

fn draw_correlation_lang_plot<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    data: &[f64],
    mean: f64,
    std_dev: f64,
    min_val: f64,
    max_val: f64,
) -> Result<()> where DB::ErrorType: 'static {
    root.fill(&WHITE)?;

    let title = format!("Распределение энтропии текстов (μ = {:.3}, σ = {:.3})", mean, std_dev);

    let mut chart = ChartBuilder::on(root)
        .caption(&title, ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_val..max_val, 0.0..1.0)?;

    chart
        .configure_mesh()
        .x_desc("Энтропия H (биты)")
        .y_desc("Нормированная частота")
        .x_label_formatter(&|v| format!("{:.2}", v))
        .y_label_formatter(&|v| format!("{:.2}", v))
        .draw()?;

    // Simple histogram visualization
    let bin_count = (data.len() as f64).sqrt().ceil() as usize;
    let bin_width = (max_val - min_val) / bin_count as f64;
    let mut bins = vec![0; bin_count];

    for &value in data {
        let bin_index = ((value - min_val) / bin_width).floor() as usize;
        if bin_index < bin_count {
            bins[bin_index] += 1;
        }
    }

    // Draw histogram bars
    for (i, &count) in bins.iter().enumerate() {
        let x_start = min_val + i as f64 * bin_width;
        let x_end = x_start + bin_width;
        let height = count as f64 / data.len() as f64;
        chart.draw_series(std::iter::once(
            Rectangle::new([(x_start, 0.0), (x_end, height)], BLUE.mix(0.5).filled())
        ))?;
    }

    Ok(())
}
