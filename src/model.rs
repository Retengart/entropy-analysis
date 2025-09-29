use serde::Serialize;

#[derive(Debug, Clone)]
pub struct Normalization {
    pub alphabet: Vec<char>,
    pub rus28_keep_yo: bool,
    pub rus28_keep_j: bool,
    pub min_token_len: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LetterRow {
    pub rank: usize,
    pub letter: char,
    pub count: u64,
    pub p: f64,
    pub p_log2: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExtendedStatsSummary {
    pub median: Option<f64>,
    pub q1: Option<f64>,
    pub q3: Option<f64>,
    pub iqr: Option<f64>,
    pub variance: Option<f64>,
    pub std_dev: Option<f64>,
    pub coefficient_of_variation: Option<f64>,
    pub skewness: Option<f64>,
    pub kurtosis: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NormalizedEntropySummary {
    pub h_normalized: Option<f64>,
    pub entropy_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummaryJson {
    pub n_words: u64,
    pub h_bits: Option<f64>,
    pub x_mean: Option<f64>,
    pub sigma: Option<f64>,
    pub non_zero_letters: usize,
    pub alphabet: String,
    pub rows: Vec<LetterRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extended_stats: Option<ExtendedStatsSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized_entropy: Option<NormalizedEntropySummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub n_words: u64,
    pub h_bits: Option<f64>,
    pub x_mean: Option<f64>,
    pub sigma: Option<f64>,
    pub non_zero_letters: usize,
    pub rows: Vec<LetterRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extended_stats: Option<ExtendedStatsSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized_entropy: Option<NormalizedEntropySummary>,
}


