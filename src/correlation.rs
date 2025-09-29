use std::path::Path;

use anyhow::Result;
use plotters::coord::Shift;
use plotters::prelude::*;

use crate::model::Summary;
use crate::cli::OutlierMethod as CliOutlierMethod;
use crate::stats::{
    calculate_extended_stats, calculate_mad, detect_outliers_iqr, 
    detect_outliers_modified_zscore, detect_outliers_zscore
};

#[derive(Debug, Clone)]
pub struct CorrelationResult {
    pub n_samples: usize,
    pub correlation_coefficient: f64,
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct FilterParams {
    pub min_entropy: Option<f64>,
    pub max_entropy: Option<f64>,
    pub min_n_words: Option<u64>,
    pub max_n_words: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct PlotParams {
    pub filter: FilterParams,
    pub label_top_entropy: Option<usize>,
    pub mark_outliers: Option<CliOutlierMethod>,
    pub outlier_threshold: Option<f64>,
}

pub struct OutlierInfo {
    pub method: CliOutlierMethod,
    pub threshold: f64,
    pub outlier_indices: Vec<usize>,
}

/// Detect outliers in the dataset
pub fn detect_outliers_in_data(
    data: &[(String, Summary)],
    method: CliOutlierMethod,
    threshold: Option<f64>,
) -> Option<OutlierInfo> {
    // Collect entropy values
    let entropy_values: Vec<f64> = data
        .iter()
        .filter_map(|(_, s)| s.h_bits)
        .collect();
    
    if entropy_values.len() < 3 {
        return None;
    }
    
    let default_threshold = match method {
        CliOutlierMethod::IQR => 1.5,
        CliOutlierMethod::ZScore => 3.0,
        CliOutlierMethod::ModifiedZScore => 3.5,
    };
    
    let threshold_value = threshold.unwrap_or(default_threshold);
    
    let outlier_indices = match method {
        CliOutlierMethod::IQR => {
            let stats = calculate_extended_stats(&entropy_values)?;
            data.iter()
                .enumerate()
                .filter_map(|(idx, (_, summary))| {
                    if let Some(h) = summary.h_bits {
                        let detection = detect_outliers_iqr(h, stats.q1, stats.q3, stats.iqr, threshold_value);
                        if detection.is_outlier {
                            Some(idx)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        },
        CliOutlierMethod::ZScore => {
            let stats = calculate_extended_stats(&entropy_values)?;
            let mean = entropy_values.iter().sum::<f64>() / entropy_values.len() as f64;
            data.iter()
                .enumerate()
                .filter_map(|(idx, (_, summary))| {
                    if let Some(h) = summary.h_bits {
                        let detection = detect_outliers_zscore(h, mean, stats.std_dev, threshold_value);
                        if detection.is_outlier {
                            Some(idx)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        },
        CliOutlierMethod::ModifiedZScore => {
            let mut sorted = entropy_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            };
            let mad = calculate_mad(&entropy_values)?;
            
            data.iter()
                .enumerate()
                .filter_map(|(idx, (_, summary))| {
                    if let Some(h) = summary.h_bits {
                        let detection = detect_outliers_modified_zscore(h, median, mad, threshold_value);
                        if detection.is_outlier {
                            Some(idx)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        },
    };
    
    Some(OutlierInfo {
        method,
        threshold: threshold_value,
        outlier_indices,
    })
}

/// Helper function to apply filters
fn apply_filters(h: f64, n_words: u64, filter: &FilterParams) -> bool {
    if let Some(min_h) = filter.min_entropy {
        if h < min_h {
            return false;
        }
    }
    if let Some(max_h) = filter.max_entropy {
        if h > max_h {
            return false;
        }
    }
    if let Some(min_n) = filter.min_n_words {
        if n_words < min_n {
            return false;
        }
    }
    if let Some(max_n) = filter.max_n_words {
        if n_words > max_n {
            return false;
        }
    }
    true
}

/// Calculate Pearson correlation coefficient between entropy H and word count N
pub fn calculate_correlation(data: &[(String, Summary)]) -> Option<CorrelationResult> {
    calculate_correlation_with_filter(data, FilterParams {
        min_entropy: None,
        max_entropy: None,
        min_n_words: None,
        max_n_words: None,
    })
}

/// Calculate correlation with optional filtering
pub fn calculate_correlation_with_filter(
    data: &[(String, Summary)],
    filter: FilterParams,
) -> Option<CorrelationResult> {
    let valid_data: Vec<(f64, f64)> = data
        .iter()
        .filter_map(|(_, summary)| {
            if let Some(h) = summary.h_bits {
                if apply_filters(h, summary.n_words, &filter) {
                    Some((summary.n_words as f64, h))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if valid_data.len() < 2 {
        return None;
    }

    let n = valid_data.len() as f64;
    
    // Calculate means
    let x_mean = valid_data.iter().map(|(x, _)| x).sum::<f64>() / n;
    let y_mean = valid_data.iter().map(|(_, y)| y).sum::<f64>() / n;

    // Calculate correlation coefficient and linear regression
    let mut numerator = 0.0;
    let mut x_variance = 0.0;
    let mut y_variance = 0.0;

    for (x, y) in &valid_data {
        let x_diff = x - x_mean;
        let y_diff = y - y_mean;
        
        numerator += x_diff * y_diff;
        x_variance += x_diff * x_diff;
        y_variance += y_diff * y_diff;
    }

    if x_variance == 0.0 || y_variance == 0.0 {
        return None;
    }

    let correlation = numerator / (x_variance * y_variance).sqrt();
    
    // Linear regression: y = slope * x + intercept
    let slope = numerator / x_variance;
    let intercept = y_mean - slope * x_mean;
    
    // R-squared
    let r_squared = correlation * correlation;

    Some(CorrelationResult {
        n_samples: valid_data.len(),
        correlation_coefficient: correlation,
        slope,
        intercept,
        r_squared,
    })
}

/// Plot scatter plot of H vs N with trend line
pub fn plot_correlation_scatter(
    path: &Path,
    data: &[(String, Summary)],
    correlation: &CorrelationResult,
) -> Result<()> {
    plot_correlation_scatter_with_params(path, data, correlation, PlotParams {
        filter: FilterParams {
            min_entropy: None,
            max_entropy: None,
            min_n_words: None,
            max_n_words: None,
        },
        label_top_entropy: None,
        mark_outliers: None,
        outlier_threshold: None,
    })
}

/// Plot scatter plot with optional filtering (deprecated - use plot_correlation_scatter_with_params)
pub fn plot_correlation_scatter_with_filter(
    path: &Path,
    data: &[(String, Summary)],
    correlation: &CorrelationResult,
    filter: FilterParams,
) -> Result<()> {
    plot_correlation_scatter_with_params(path, data, correlation, PlotParams {
        filter,
        label_top_entropy: None,
        mark_outliers: None,
        outlier_threshold: None,
    })
}

/// Plot scatter plot with filtering and labeling options
pub fn plot_correlation_scatter_with_params(
    path: &Path,
    data: &[(String, Summary)],
    correlation: &CorrelationResult,
    params: PlotParams,
) -> Result<()> {
    // Collect data with labels (with original indices)
    let labeled_data_with_idx: Vec<(usize, String, f64, f64)> = data
        .iter()
        .enumerate()
        .filter_map(|(idx, (name, summary))| {
            if let Some(h) = summary.h_bits {
                if apply_filters(h, summary.n_words, &params.filter) {
                    Some((idx, name.clone(), summary.n_words as f64, h))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if labeled_data_with_idx.is_empty() {
        return Ok(());
    }
    
    // Detect outliers if requested
    let outlier_info = if let Some(method) = params.mark_outliers {
        detect_outliers_in_data(data, method, params.outlier_threshold)
    } else {
        None
    };
    
    // Determine which texts to label
    let texts_to_label: Vec<String> = if let Some(top_n) = params.label_top_entropy {
        let mut sorted_by_entropy = labeled_data_with_idx.clone();
        sorted_by_entropy.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        sorted_by_entropy.iter()
            .take(top_n)
            .map(|(_, name, _, _)| name.clone())
            .collect()
    } else {
        Vec::new()
    };
    
    let labeled_data: Vec<(String, f64, f64)> = labeled_data_with_idx.iter()
        .map(|(_, name, n, h)| (name.clone(), *n, *h))
        .collect();
    
    // Create outlier set for quick lookup
    let outlier_set: std::collections::HashSet<usize> = if let Some(ref info) = outlier_info {
        info.outlier_indices.iter().copied().collect()
    } else {
        std::collections::HashSet::new()
    };

    let x_min = labeled_data.iter().map(|(_, x, _)| *x).fold(f64::INFINITY, |a, b| a.min(b));
    let x_max = labeled_data.iter().map(|(_, x, _)| *x).fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let y_min = labeled_data.iter().map(|(_, _, y)| *y).fold(f64::INFINITY, |a, b| a.min(b));
    let y_max = labeled_data.iter().map(|(_, _, y)| *y).fold(f64::NEG_INFINITY, |a, b| a.max(b));

    // Add some padding
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    let x_padding = x_range * 0.1;
    let y_padding = y_range * 0.1;

    let (is_svg, path_str) = if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        (ext.eq_ignore_ascii_case("svg"), path.to_string_lossy().to_string())
    } else {
        (false, path.to_string_lossy().to_string())
    };

    if is_svg {
        let root = SVGBackend::new(&path_str, (1200, 800)).into_drawing_area();
        draw_correlation_plot(&root, &labeled_data_with_idx, &texts_to_label, &outlier_set, correlation, 
                             x_min - x_padding, x_max + x_padding, 
                             y_min - y_padding, y_max + y_padding)?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(&path_str, (1200, 800)).into_drawing_area();
        draw_correlation_plot(&root, &labeled_data_with_idx, &texts_to_label, &outlier_set, correlation,
                             x_min - x_padding, x_max + x_padding, 
                             y_min - y_padding, y_max + y_padding)?;
        root.present()?;
    }
    Ok(())
}

fn draw_correlation_plot<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    labeled_data_with_idx: &[(usize, String, f64, f64)],
    texts_to_label: &[String],
    outlier_indices: &std::collections::HashSet<usize>,
    correlation: &CorrelationResult,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Result<()> where DB::ErrorType: 'static {
    root.fill(&WHITE)?;
    
    let title = format!("Correlation: H vs N (r = {:.4}, R² = {:.4}), n = {}", 
                       correlation.correlation_coefficient, correlation.r_squared, labeled_data_with_idx.len());
    
    let mut chart = ChartBuilder::on(root)
        .caption(&title, ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("N (количество слов)")
        .y_desc("H (энтропия, бит)")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.2}", v))
        .draw()?;

    // Draw data points with outlier highlighting
    for (idx, _name, x, y) in labeled_data_with_idx.iter() {
        let is_outlier = outlier_indices.contains(idx);
        let color = if is_outlier { MAGENTA } else { BLUE };
        let size = if is_outlier { 5 } else { 4 };
        chart.draw_series(std::iter::once(Circle::new((*x, *y), size, color.filled())))?;
    }
    
    // Draw labels for top entropy texts
    if !texts_to_label.is_empty() {
        for (_idx, name, x, y) in labeled_data_with_idx.iter() {
            if texts_to_label.contains(name) {
                // Extract just the filename or last part and make owned string
                let label = name.split('/').last().unwrap_or(name)
                    .split('\\').last().unwrap_or(name)
                    .to_string();
                
                chart.draw_series(std::iter::once(Text::new(
                    label.clone(),
                    (*x, *y + (y_max - y_min) * 0.02),
                    ("sans-serif", 12).into_font().color(&RED),
                )))?;
                
                // Highlight labeled points in red
                chart.draw_series(std::iter::once(Circle::new((*x, *y), 6, RED.filled())))?;
            }
        }
    }

    // Draw trend line
    let trend_start_y = correlation.slope * x_min + correlation.intercept;
    let trend_end_y = correlation.slope * x_max + correlation.intercept;
    
    chart.draw_series(LineSeries::new(
        vec![(x_min, trend_start_y), (x_max, trend_end_y)],
        RED.stroke_width(2),
    ))?;

    // Add equation text
    let equation = format!("y = {:.4}x + {:.4}", correlation.slope, correlation.intercept);
    chart.draw_series(std::iter::once(Text::new(
        equation,
        (x_min + (x_max - x_min) * 0.05, y_max - (y_max - y_min) * 0.1),
        ("sans-serif", 20).into_font(),
    )))?;

    Ok(())
}

/// Plot scatter plot with two authors in different colors and separate trend lines
pub fn plot_dual_author_correlation(
    path: &Path,
    data1: &[(String, Summary)],
    data2: &[(String, Summary)],
    author1_name: &str,
    author2_name: &str,
) -> Result<()> {
    plot_dual_author_correlation_with_filter(
        path, data1, data2, author1_name, author2_name,
        FilterParams {
            min_entropy: None,
            max_entropy: None,
            min_n_words: None,
            max_n_words: None,
        }
    )
}

/// Plot dual author correlation with optional filtering
pub fn plot_dual_author_correlation_with_filter(
    path: &Path,
    data1: &[(String, Summary)],
    data2: &[(String, Summary)],
    author1_name: &str,
    author2_name: &str,
    filter: FilterParams,
) -> Result<()> {
    let valid_data1: Vec<(f64, f64)> = data1
        .iter()
        .filter_map(|(_, summary)| {
            if let Some(h) = summary.h_bits {
                if apply_filters(h, summary.n_words, &filter) {
                    Some((summary.n_words as f64, h))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    let valid_data2: Vec<(f64, f64)> = data2
        .iter()
        .filter_map(|(_, summary)| {
            if let Some(h) = summary.h_bits {
                if apply_filters(h, summary.n_words, &filter) {
                    Some((summary.n_words as f64, h))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if valid_data1.is_empty() && valid_data2.is_empty() {
        return Ok(());
    }

    // Calculate ranges from both datasets
    let all_x: Vec<f64> = valid_data1.iter().chain(valid_data2.iter()).map(|(x, _)| *x).collect();
    let all_y: Vec<f64> = valid_data1.iter().chain(valid_data2.iter()).map(|(_, y)| *y).collect();
    
    let x_min = all_x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = all_x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = all_y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = all_y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Add some padding
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    let x_padding = x_range * 0.1;
    let y_padding = y_range * 0.1;

    // Calculate correlations for both authors
    let corr1 = calculate_correlation(data1);
    let corr2 = calculate_correlation(data2);

    let (is_svg, path_str) = if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        (ext.eq_ignore_ascii_case("svg"), path.to_string_lossy().to_string())
    } else {
        (false, path.to_string_lossy().to_string())
    };

    if is_svg {
        let root = SVGBackend::new(&path_str, (1200, 800)).into_drawing_area();
        draw_dual_author_plot(&root, &valid_data1, &valid_data2, 
                             author1_name, author2_name,
                             corr1.as_ref(), corr2.as_ref(),
                             x_min - x_padding, x_max + x_padding, 
                             y_min - y_padding, y_max + y_padding)?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(&path_str, (1200, 800)).into_drawing_area();
        draw_dual_author_plot(&root, &valid_data1, &valid_data2, 
                             author1_name, author2_name,
                             corr1.as_ref(), corr2.as_ref(),
                             x_min - x_padding, x_max + x_padding, 
                             y_min - y_padding, y_max + y_padding)?;
        root.present()?;
    }
    Ok(())
}

fn draw_dual_author_plot<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    data1: &[(f64, f64)],
    data2: &[(f64, f64)],
    author1_name: &str,
    author2_name: &str,
    corr1: Option<&CorrelationResult>,
    corr2: Option<&CorrelationResult>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Result<()> where DB::ErrorType: 'static {
    root.fill(&WHITE)?;
    
    let title = format!("Correlation Comparison: {} vs {}", author1_name, author2_name);
    
    let mut chart = ChartBuilder::on(root)
        .caption(&title, ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("N (количество слов)")
        .y_desc("H (энтропия, бит)")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.2}", v))
        .draw()?;

    // Draw data points for author 1 (blue)
    chart.draw_series(data1.iter().map(|(x, y)| Circle::new((*x, *y), 4, BLUE.filled())))?
        .label(&format!("{} (n={})", author1_name, data1.len()))
        .legend(|(x, y)| Circle::new((x + 10, y), 4, BLUE.filled()));

    // Draw data points for author 2 (red)
    chart.draw_series(data2.iter().map(|(x, y)| Circle::new((*x, *y), 4, RED.filled())))?
        .label(&format!("{} (n={})", author2_name, data2.len()))
        .legend(|(x, y)| Circle::new((x + 10, y), 4, RED.filled()));

    // Draw trend line for author 1
    if let Some(correlation1) = corr1 {
        let trend_start_y = correlation1.slope * x_min + correlation1.intercept;
        let trend_end_y = correlation1.slope * x_max + correlation1.intercept;
        
        chart.draw_series(LineSeries::new(
            vec![(x_min, trend_start_y), (x_max, trend_end_y)],
            BLUE.stroke_width(2),
        ))?
        .label(&format!("{} (r={:.3})", author1_name, correlation1.correlation_coefficient))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], BLUE.stroke_width(2)));
    }

    // Draw trend line for author 2
    if let Some(correlation2) = corr2 {
        let trend_start_y = correlation2.slope * x_min + correlation2.intercept;
        let trend_end_y = correlation2.slope * x_max + correlation2.intercept;
        
        chart.draw_series(LineSeries::new(
            vec![(x_min, trend_start_y), (x_max, trend_end_y)],
            RED.stroke_width(2),
        ))?
        .label(&format!("{} (r={:.3})", author2_name, correlation2.correlation_coefficient))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], RED.stroke_width(2)));
    }

    // Add equations
    let mut y_offset = 0.1;
    if let Some(correlation1) = corr1 {
        let equation1 = format!("{}: y = {:.4}x + {:.4}", author1_name, correlation1.slope, correlation1.intercept);
        chart.draw_series(std::iter::once(Text::new(
            equation1,
            (x_min + (x_max - x_min) * 0.05, y_max - (y_max - y_min) * y_offset),
            ("sans-serif", 16).into_font().color(&BLUE),
        )))?;
        y_offset += 0.08;
    }
    
    if let Some(correlation2) = corr2 {
        let equation2 = format!("{}: y = {:.4}x + {:.4}", author2_name, correlation2.slope, correlation2.intercept);
        chart.draw_series(std::iter::once(Text::new(
            equation2,
            (x_min + (x_max - x_min) * 0.05, y_max - (y_max - y_min) * y_offset),
            ("sans-serif", 16).into_font().color(&RED),
        )))?;
    }

    chart.configure_series_labels().draw()?;

    Ok(())
}
