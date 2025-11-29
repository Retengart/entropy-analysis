use std::fs;
use std::io::{self, Read, Write};

use anyhow::Result;
use clap::Parser;

mod analyze;
mod cli;
mod correlation;
mod model;
mod multi_poem;
mod normalize;
mod output;
mod plot;
mod services;
mod stats;

use analyze::{add_normalized_entropy, analyze_text, calculate_batch_extended_stats};
use cli::{AnalyzeArgs, BatchArgs, Cli, Commands, MultiPoemArgs};
use correlation::{
    calculate_correlation_with_filter, detect_outliers_in_data,
    plot_correlation_scatter_with_params, plot_dual_author_correlation_with_filter, FilterParams,
    PlotParams,
};
use multi_poem::{
    analyze_multi_poem, analyze_two_authors, create_analysis_table, plot_correlation_lang,
    plot_normal_distribution, print_analysis_table, write_analysis_table_csv, TwoAuthorsData,
};
use normalize::build_normalization;
use output::{print_console_summary, write_csv, write_json, write_summary_csv, write_summary_json};
use plot::plot_histogram;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Analyze(args) => run_analyze(args),
        Commands::Batch(args) => run_batch(args),
        Commands::MultiPoem(args) => run_multi_poem(args),
    }
}

fn run_analyze(args: AnalyzeArgs) -> Result<()> {
    let text = read_input(&args.input)?;
    let norm = build_normalization(
        args.alphabet,
        args.custom_alphabet.as_deref(),
        args.keep_yo,
        args.keep_j,
        args.min_token_len,
    )?;

    let summary = analyze_text(&text, &norm)?;
    print_console_summary(&summary)?;

    if let Some(path) = args.csv.as_deref() {
        write_csv(path, &summary)?;
    }
    if let Some(path) = args.json.as_deref() {
        write_json(path, &summary, &norm)?;
    }
    if let Some(path) = args.hist_alpha.as_deref() {
        plot_histogram(path, &summary.rows, false)?;
    }
    if let Some(path) = args.hist_sorted.as_deref() {
        let mut rows = summary.rows.clone();
        rows.sort_by(|a, b| a.p.partial_cmp(&b.p).unwrap_or(std::cmp::Ordering::Equal));
        plot_histogram(path, &rows, true)?;
    }
    Ok(())
}

fn run_batch(args: BatchArgs) -> Result<()> {
    let norm = build_normalization(
        args.alphabet,
        args.custom_alphabet.as_deref(),
        args.keep_yo,
        args.keep_j,
        1,
    )?;
    let mut rows: Vec<(String, model::Summary)> = Vec::new();
    let walker = walkdir::WalkDir::new(&args.dir)
        .max_depth(if args.recurse { usize::MAX } else { 1 })
        .into_iter()
        .filter_map(|e| e.ok());
    for entry in walker {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()).unwrap_or("") != args.ext {
            continue;
        }
        let text = fs::read_to_string(path)?;
        let mut summary = analyze_text(&text, &norm)?;

        // Add normalized entropy if requested
        if args.normalized_entropy || args.entropy_rate {
            add_normalized_entropy(&mut summary);
        }

        rows.push((path.to_string_lossy().to_string(), summary));
    }

    // console output
    let mut stdout = io::stdout();

    // Print extended statistics if requested
    if args.show_extended_stats {
        if let Some(ext_stats) = calculate_batch_extended_stats(&rows) {
            writeln!(stdout, "\n=== Extended Statistics (Entropy H) ===")?;
            writeln!(
                stdout,
                "Median:      {:.6}",
                ext_stats.median.unwrap_or(0.0)
            )?;
            writeln!(stdout, "Q1 (25%):    {:.6}", ext_stats.q1.unwrap_or(0.0))?;
            writeln!(stdout, "Q3 (75%):    {:.6}", ext_stats.q3.unwrap_or(0.0))?;
            writeln!(stdout, "IQR:         {:.6}", ext_stats.iqr.unwrap_or(0.0))?;
            writeln!(
                stdout,
                "Std Dev:     {:.6}",
                ext_stats.std_dev.unwrap_or(0.0)
            )?;
            writeln!(
                stdout,
                "Variance:    {:.6}",
                ext_stats.variance.unwrap_or(0.0)
            )?;
            writeln!(
                stdout,
                "CV (%):      {:.2}",
                ext_stats.coefficient_of_variation.unwrap_or(0.0)
            )?;
            writeln!(
                stdout,
                "Skewness:    {:.6}",
                ext_stats.skewness.unwrap_or(0.0)
            )?;
            writeln!(
                stdout,
                "Kurtosis:    {:.6}",
                ext_stats.kurtosis.unwrap_or(0.0)
            )?;
            writeln!(stdout, "")?;
        }
    }

    // Detect outliers if requested
    if let Some(method) = args.auto_filter_outliers {
        if let Some(outlier_info) = detect_outliers_in_data(&rows, method, args.outlier_threshold) {
            writeln!(stdout, "\n=== Outlier Detection ({:?} method) ===", method)?;
            writeln!(stdout, "Threshold: {:.2}", outlier_info.threshold)?;
            writeln!(
                stdout,
                "Outliers found: {}",
                outlier_info.outlier_indices.len()
            )?;
            if !outlier_info.outlier_indices.is_empty() {
                writeln!(stdout, "\nOutlier files:")?;
                for &idx in &outlier_info.outlier_indices {
                    if let Some((file, summary)) = rows.get(idx) {
                        writeln!(
                            stdout,
                            "  {} (H = {:.4})",
                            file,
                            summary.h_bits.unwrap_or(0.0)
                        )?;
                    }
                }
            }
            writeln!(stdout, "")?;
        }
    }

    writeln!(stdout, "file,N,H_bits,x_mean,sigma")?;
    for (file, s) in &rows {
        writeln!(
            stdout,
            "{},{},{},{},{}",
            file,
            s.n_words,
            fmt_opt(s.h_bits),
            fmt_opt(s.x_mean),
            fmt_opt(s.sigma)
        )?;
    }

    // correlation analysis
    if args.correlation || args.correlation_plot.is_some() {
        let params = PlotParams {
            filter: FilterParams {
                min_entropy: args.min_entropy,
                max_entropy: args.max_entropy,
                min_n_words: args.min_n_words,
                max_n_words: args.max_n_words,
            },
            label_top_entropy: args.label_top_entropy,
            mark_outliers: args.mark_outliers,
            outlier_threshold: args.outlier_threshold,
        };

        if let Some(corr_result) = calculate_correlation_with_filter(&rows, params.filter) {
            writeln!(stdout, "\nCorrelation Analysis:")?;
            if args.min_entropy.is_some()
                || args.max_entropy.is_some()
                || args.min_n_words.is_some()
                || args.max_n_words.is_some()
            {
                writeln!(stdout, "Filters applied: min_entropy={:?}, max_entropy={:?}, min_n_words={:?}, max_n_words={:?}",
                    args.min_entropy, args.max_entropy, args.min_n_words, args.max_n_words)?;
            }
            if args.label_top_entropy.is_some() {
                writeln!(
                    stdout,
                    "Labeling top {} texts by entropy",
                    args.label_top_entropy.unwrap()
                )?;
            }
            writeln!(stdout, "Samples: {}", corr_result.n_samples)?;
            writeln!(
                stdout,
                "Correlation coefficient (r): {:.6}",
                corr_result.correlation_coefficient
            )?;
            writeln!(stdout, "R-squared: {:.6}", corr_result.r_squared)?;
            writeln!(
                stdout,
                "Linear regression: H = {:.6} * N + {:.6}",
                corr_result.slope, corr_result.intercept
            )?;

            if let Some(plot_path) = args.correlation_plot.as_deref() {
                plot_correlation_scatter_with_params(plot_path, &rows, &corr_result, params)?;
                writeln!(stdout, "Correlation plot saved to: {}", plot_path.display())?;
            }
        } else {
            writeln!(
                stdout,
                "\nWarning: Could not calculate correlation (insufficient valid data)"
            )?;
        }
    }

    // file outputs
    if let Some(p) = args.summary_csv.as_deref() {
        write_summary_csv(p, &rows)?;
    }
    if let Some(p) = args.summary_json.as_deref() {
        write_summary_json(p, &rows)?;
    }
    Ok(())
}

fn read_input(input: &str) -> Result<String> {
    if input == "-" {
        let mut s = String::new();
        io::stdin().read_to_string(&mut s)?;
        Ok(s)
    } else {
        Ok(fs::read_to_string(input)?)
    }
}

fn run_multi_poem(args: MultiPoemArgs) -> Result<()> {
    let text = read_input(&args.input)?;
    let norm = build_normalization(
        args.alphabet,
        args.custom_alphabet.as_deref(),
        args.keep_yo,
        args.keep_j,
        1,
    )?;

    // Check if we have two authors or just one
    if let Some(second_file) = &args.second_file {
        let text2 = read_input(second_file)?;
        let two_authors =
            analyze_two_authors(&text, &text2, &norm, &args.delimiter, "Автор 1", "Автор 2")?;

        // Print analysis tables for both authors
        let analysis_table1 = create_analysis_table(&two_authors.author1);
        let analysis_table2 = create_analysis_table(&two_authors.author2);

        let mut stdout = io::stdout();
        writeln!(
            stdout,
            "\nАнализ {} (файл: {}):",
            two_authors.author1_name, args.input
        )?;
        print_analysis_table(&analysis_table1)?;

        writeln!(
            stdout,
            "\nАнализ {} (файл: {}):",
            two_authors.author2_name, second_file
        )?;
        print_analysis_table(&analysis_table2)?;

        // Handle two-author correlation plotting
        run_two_author_correlation(&args, &two_authors)?;
    } else {
        // Single author mode (existing logic)
        let poems = analyze_multi_poem(&text, &norm, &args.delimiter)?;
        let analysis_table = create_analysis_table(&poems);

        // Print analysis table to console
        print_analysis_table(&analysis_table)?;

        // Convert to format compatible with batch analysis
        let rows: Vec<(String, model::Summary)> = poems
            .iter()
            .map(|poem| (poem.title.clone(), poem.summary.clone()))
            .collect();

        run_single_author_analysis(&args, &rows, &analysis_table, &poems)?;
    }

    Ok(())
}

fn run_single_author_analysis(
    args: &MultiPoemArgs,
    rows: &[(String, model::Summary)],
    analysis_table: &[multi_poem::AnalysisTableRow],
    poems: &[multi_poem::PoemSummary],
) -> Result<()> {
    // Console output summary
    let mut stdout = io::stdout();
    writeln!(stdout, "\nSummary:")?;
    writeln!(stdout, "file,N,H_bits,x_mean,sigma")?;
    for (title, s) in rows {
        writeln!(
            stdout,
            "{},{},{},{},{}",
            title,
            s.n_words,
            fmt_opt(s.h_bits),
            fmt_opt(s.x_mean),
            fmt_opt(s.sigma)
        )?;
    }

    // Correlation analysis
    if args.correlation || args.correlation_plot.is_some() {
        let params = PlotParams {
            filter: FilterParams {
                min_entropy: args.min_entropy,
                max_entropy: args.max_entropy,
                min_n_words: args.min_n_words,
                max_n_words: args.max_n_words,
            },
            label_top_entropy: args.label_top_entropy,
            mark_outliers: args.mark_outliers,
            outlier_threshold: args.outlier_threshold,
        };

        if let Some(corr_result) = calculate_correlation_with_filter(rows, params.filter) {
            writeln!(stdout, "\nCorrelation Analysis:")?;
            if args.min_entropy.is_some()
                || args.max_entropy.is_some()
                || args.min_n_words.is_some()
                || args.max_n_words.is_some()
            {
                writeln!(stdout, "Filters applied: min_entropy={:?}, max_entropy={:?}, min_n_words={:?}, max_n_words={:?}",
                    args.min_entropy, args.max_entropy, args.min_n_words, args.max_n_words)?;
            }
            if args.label_top_entropy.is_some() {
                writeln!(
                    stdout,
                    "Labeling top {} texts by entropy",
                    args.label_top_entropy.unwrap()
                )?;
            }
            writeln!(stdout, "Samples: {}", corr_result.n_samples)?;
            writeln!(
                stdout,
                "Correlation coefficient (r): {:.6}",
                corr_result.correlation_coefficient
            )?;
            writeln!(stdout, "R-squared: {:.6}", corr_result.r_squared)?;
            writeln!(
                stdout,
                "Linear regression: H = {:.6} * N + {:.6}",
                corr_result.slope, corr_result.intercept
            )?;

            if let Some(plot_path) = args.correlation_plot.as_deref() {
                plot_correlation_scatter_with_params(plot_path, rows, &corr_result, params)?;
                writeln!(stdout, "Correlation plot saved to: {}", plot_path.display())?;
            }
        } else {
            writeln!(
                stdout,
                "\nWarning: Could not calculate correlation (insufficient valid data)"
            )?;
        }
    }

    // Normal distribution plot
    if let Some(normal_plot_path) = args.normal_dist_plot.as_deref() {
        plot_normal_distribution(normal_plot_path, poems)?;
        writeln!(
            stdout,
            "Normal distribution plot saved to: {}",
            normal_plot_path.display()
        )?;
    }

    // Correlation between translation systems plot
    if let Some(correlation_lang_path) = args.correlation_lang.as_deref() {
        plot_correlation_lang(correlation_lang_path, poems)?;
        writeln!(
            stdout,
            "Translation systems correlation plot saved to: {}",
            correlation_lang_path.display()
        )?;
    }

    // File outputs
    if let Some(p) = args.analysis_table.as_deref() {
        write_analysis_table_csv(p, analysis_table)?;
        writeln!(stdout, "Analysis table saved to: {}", p.display())?;
    }
    if let Some(p) = args.summary_csv.as_deref() {
        write_summary_csv(p, rows)?;
    }
    if let Some(p) = args.summary_json.as_deref() {
        write_summary_json(p, rows)?;
    }

    Ok(())
}

fn run_two_author_correlation(args: &MultiPoemArgs, two_authors: &TwoAuthorsData) -> Result<()> {
    let mut stdout = io::stdout();

    // Convert both authors' data to format compatible with correlation analysis
    let rows1: Vec<(String, model::Summary)> = two_authors
        .author1
        .iter()
        .map(|poem| {
            (
                format!("{} - {}", two_authors.author1_name, poem.title),
                poem.summary.clone(),
            )
        })
        .collect();

    let rows2: Vec<(String, model::Summary)> = two_authors
        .author2
        .iter()
        .map(|poem| {
            (
                format!("{} - {}", two_authors.author2_name, poem.title),
                poem.summary.clone(),
            )
        })
        .collect();

    // Console output summary for both authors
    writeln!(stdout, "\nSummary for {}:", two_authors.author1_name)?;
    writeln!(stdout, "file,N,H_bits,x_mean,sigma")?;
    for (title, s) in &rows1 {
        writeln!(
            stdout,
            "{},{},{},{},{}",
            title,
            s.n_words,
            fmt_opt(s.h_bits),
            fmt_opt(s.x_mean),
            fmt_opt(s.sigma)
        )?;
    }

    writeln!(stdout, "\nSummary for {}:", two_authors.author2_name)?;
    writeln!(stdout, "file,N,H_bits,x_mean,sigma")?;
    for (title, s) in &rows2 {
        writeln!(
            stdout,
            "{},{},{},{},{}",
            title,
            s.n_words,
            fmt_opt(s.h_bits),
            fmt_opt(s.x_mean),
            fmt_opt(s.sigma)
        )?;
    }

    let params = PlotParams {
        filter: FilterParams {
            min_entropy: args.min_entropy,
            max_entropy: args.max_entropy,
            min_n_words: args.min_n_words,
            max_n_words: args.max_n_words,
        },
        label_top_entropy: args.label_top_entropy,
        mark_outliers: args.mark_outliers,
        outlier_threshold: args.outlier_threshold,
    };

    // Individual correlation analysis
    if args.correlation {
        if args.min_entropy.is_some()
            || args.max_entropy.is_some()
            || args.min_n_words.is_some()
            || args.max_n_words.is_some()
        {
            writeln!(stdout, "\nFilters applied: min_entropy={:?}, max_entropy={:?}, min_n_words={:?}, max_n_words={:?}",
                args.min_entropy, args.max_entropy, args.min_n_words, args.max_n_words)?;
        }
        if args.label_top_entropy.is_some() {
            writeln!(
                stdout,
                "Labeling top {} texts by entropy",
                args.label_top_entropy.unwrap()
            )?;
        }

        if let Some(corr_result1) = calculate_correlation_with_filter(&rows1, params.filter) {
            writeln!(
                stdout,
                "\nCorrelation Analysis for {}:",
                two_authors.author1_name
            )?;
            writeln!(stdout, "Samples: {}", corr_result1.n_samples)?;
            writeln!(
                stdout,
                "Correlation coefficient (r): {:.6}",
                corr_result1.correlation_coefficient
            )?;
            writeln!(stdout, "R-squared: {:.6}", corr_result1.r_squared)?;
            writeln!(
                stdout,
                "Linear regression: H = {:.6} * N + {:.6}",
                corr_result1.slope, corr_result1.intercept
            )?;
        }

        if let Some(corr_result2) = calculate_correlation_with_filter(&rows2, params.filter) {
            writeln!(
                stdout,
                "\nCorrelation Analysis for {}:",
                two_authors.author2_name
            )?;
            writeln!(stdout, "Samples: {}", corr_result2.n_samples)?;
            writeln!(
                stdout,
                "Correlation coefficient (r): {:.6}",
                corr_result2.correlation_coefficient
            )?;
            writeln!(stdout, "R-squared: {:.6}", corr_result2.r_squared)?;
            writeln!(
                stdout,
                "Linear regression: H = {:.6} * N + {:.6}",
                corr_result2.slope, corr_result2.intercept
            )?;
        }
    }

    // Combined dual-author correlation plot
    if let Some(plot_path) = args.correlation_plot.as_deref() {
        plot_dual_author_correlation_with_filter(
            plot_path,
            &rows1,
            &rows2,
            &two_authors.author1_name,
            &two_authors.author2_name,
            params.filter,
        )?;
        writeln!(
            stdout,
            "Dual-author correlation plot saved to: {}",
            plot_path.display()
        )?;
    }

    Ok(())
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{:.6}", x)).unwrap_or_default()
}
