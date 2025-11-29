use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(name = "entropy-analysis", version, about = "Entropy analysis of poems by initial letter")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Analyze(AnalyzeArgs),
    Batch(BatchArgs),
    MultiPoem(MultiPoemArgs),
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum AlphabetChoice {
    Rus28,
    Rus33,
    Custom,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq)]
pub enum OutlierMethod {
    /// Interquartile Range method (default threshold: 1.5)
    IQR,
    /// Z-score method (default threshold: 3.0)
    ZScore,
    /// Modified Z-score using MAD (default threshold: 3.5)
    ModifiedZScore,
}

#[derive(Args, Debug)]
pub struct AnalyzeArgs {
    /// Input file path or '-' for stdin
    pub input: String,

    /// Alphabet type: rus28 (default), rus33 or custom
    #[arg(long, value_enum, default_value_t = AlphabetChoice::Rus28)]
    pub alphabet: AlphabetChoice,

    /// Custom alphabet (used when --alphabet=custom). Order defines ranks.
    #[arg(long)]
    pub custom_alphabet: Option<String>,

    /// Keep 'ё' as separate letter (rus28 mode only). By default 'ё'→'е'.
    #[arg(long)]
    pub keep_yo: bool,

    /// Keep 'й' as separate letter (rus28 mode only). By default 'й'→'и'.
    #[arg(long)]
    pub keep_j: bool,

    /// Minimum token length to consider as word
    #[arg(long, default_value_t = 1)]
    pub min_token_len: usize,

    /// Write per-letter CSV table
    #[arg(long)]
    pub csv: Option<std::path::PathBuf>,

    /// Write JSON summary
    #[arg(long)]
    pub json: Option<std::path::PathBuf>,

    /// PNG/SVG histogram in alphabet order
    #[arg(long)]
    pub hist_alpha: Option<std::path::PathBuf>,

    /// PNG/SVG histogram sorted by probability
    #[arg(long)]
    pub hist_sorted: Option<std::path::PathBuf>,
}

#[derive(Args, Debug)]
pub struct BatchArgs {
    /// Directory with text files
    pub dir: std::path::PathBuf,

    /// Recurse into subdirectories
    #[arg(long)]
    pub recurse: bool,

    /// File extension to include, e.g. txt
    #[arg(long, default_value = "txt")]
    pub ext: String,

    /// Summary CSV output path
    #[arg(long)]
    pub summary_csv: Option<std::path::PathBuf>,

    /// Summary JSON output path
    #[arg(long)]
    pub summary_json: Option<std::path::PathBuf>,

    /// Calculate and display correlation coefficient between H and N
    #[arg(long)]
    pub correlation: bool,

    /// Generate correlation scatter plot H vs N with trend line
    #[arg(long)]
    pub correlation_plot: Option<std::path::PathBuf>,

    /// Minimum entropy (H) threshold for filtering points on plots
    #[arg(long)]
    pub min_entropy: Option<f64>,

    /// Maximum entropy (H) threshold for filtering points on plots
    #[arg(long)]
    pub max_entropy: Option<f64>,

    /// Minimum word count (N) threshold for filtering points on plots
    #[arg(long)]
    pub min_n_words: Option<u64>,

    /// Maximum word count (N) threshold for filtering points on plots
    #[arg(long)]
    pub max_n_words: Option<u64>,

    /// Label N texts with highest entropy on correlation plots
    #[arg(long)]
    pub label_top_entropy: Option<usize>,

    /// Show extended statistical metrics (median, quartiles, IQR, skewness, kurtosis)
    #[arg(long)]
    pub show_extended_stats: bool,

    /// Use normalized entropy (H / log2(N)) instead of raw entropy
    #[arg(long)]
    pub normalized_entropy: bool,

    /// Show entropy rate (H / N)
    #[arg(long)]
    pub entropy_rate: bool,

    /// Automatically filter outliers using specified method
    #[arg(long)]
    pub auto_filter_outliers: Option<OutlierMethod>,

    /// Mark outliers on plots with different color (without filtering them)
    #[arg(long)]
    pub mark_outliers: Option<OutlierMethod>,

    /// Outlier detection threshold (default: 1.5 for IQR, 3.0 for Z-score, 3.5 for ModifiedZScore)
    #[arg(long)]
    pub outlier_threshold: Option<f64>,

    #[arg(long, value_enum, default_value_t = AlphabetChoice::Rus28)]
    pub alphabet: AlphabetChoice,

    #[arg(long)]
    pub custom_alphabet: Option<String>,

    #[arg(long)]
    pub keep_yo: bool,

    #[arg(long)]
    pub keep_j: bool,
}

#[derive(Args, Debug)]
pub struct MultiPoemArgs {
    /// Input file with multiple poems separated by custom delimiter
    pub input: String,

    /// Optional second file with poems from another author
    #[arg(long)]
    pub second_file: Option<String>,

    /// Delimiter for separating poems (default: "***")
    #[arg(long, default_value = "***")]
    pub delimiter: String,

    /// Output detailed analysis table (like in the methodology)
    #[arg(long)]
    pub analysis_table: Option<std::path::PathBuf>,

    /// Summary CSV output path
    #[arg(long)]
    pub summary_csv: Option<std::path::PathBuf>,

    /// Summary JSON output path
    #[arg(long)]
    pub summary_json: Option<std::path::PathBuf>,

    /// Calculate and display correlation coefficient between H and N
    #[arg(long)]
    pub correlation: bool,

    /// Generate correlation scatter plot H vs N with trend line
    #[arg(long)]
    pub correlation_plot: Option<std::path::PathBuf>,

    /// Generate normal distribution plot for entropy values
    #[arg(long)]
    pub normal_dist_plot: Option<std::path::PathBuf>,

    /// Generate correlation plot comparing different translation systems
    #[arg(long)]
    pub correlation_lang: Option<std::path::PathBuf>,

    /// Minimum entropy (H) threshold for filtering points on plots
    #[arg(long)]
    pub min_entropy: Option<f64>,

    /// Maximum entropy (H) threshold for filtering points on plots
    #[arg(long)]
    pub max_entropy: Option<f64>,

    /// Minimum word count (N) threshold for filtering points on plots
    #[arg(long)]
    pub min_n_words: Option<u64>,

    /// Maximum word count (N) threshold for filtering points on plots
    #[arg(long)]
    pub max_n_words: Option<u64>,

    /// Label N texts with highest entropy on correlation plots
    #[arg(long)]
    pub label_top_entropy: Option<usize>,

    /// Show extended statistical metrics (median, quartiles, IQR, skewness, kurtosis)
    #[arg(long)]
    pub show_extended_stats: bool,

    /// Use normalized entropy (H / log2(N)) instead of raw entropy
    #[arg(long)]
    pub normalized_entropy: bool,

    /// Show entropy rate (H / N)
    #[arg(long)]
    pub entropy_rate: bool,

    /// Automatically filter outliers using specified method
    #[arg(long)]
    pub auto_filter_outliers: Option<OutlierMethod>,

    /// Mark outliers on plots with different color (without filtering them)
    #[arg(long)]
    pub mark_outliers: Option<OutlierMethod>,

    /// Outlier detection threshold (default: 1.5 for IQR, 3.0 for Z-score, 3.5 for ModifiedZScore)
    #[arg(long)]
    pub outlier_threshold: Option<f64>,

    #[arg(long, value_enum, default_value_t = AlphabetChoice::Rus28)]
    pub alphabet: AlphabetChoice,

    #[arg(long)]
    pub custom_alphabet: Option<String>,

    #[arg(long)]
    pub keep_yo: bool,

    #[arg(long)]
    pub keep_j: bool,
}


