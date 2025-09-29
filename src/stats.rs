/// Advanced statistical functions for entropy analysis
use std::f64::consts::E;

#[derive(Debug, Clone, Copy)]
pub struct ExtendedStats {
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub coefficient_of_variation: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct NormalizedEntropy {
    pub h_norm: f64,        // H / log2(N)
    pub entropy_rate: f64,   // H / N
}

#[derive(Debug, Clone, Copy)]
pub struct SaturationModel {
    pub h_max: f64,
    pub n0: f64,
    pub r_squared: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutlierMethod {
    IQR,
    ZScore,
    ModifiedZScore,
}

#[derive(Debug, Clone)]
pub struct OutlierDetection {
    pub is_outlier: bool,
    pub score: f64,
    pub method: OutlierMethod,
}

/// Calculate median from sorted values
pub fn median(sorted_values: &[f64]) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    
    let len = sorted_values.len();
    if len % 2 == 0 {
        Some((sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0)
    } else {
        Some(sorted_values[len / 2])
    }
}

/// Calculate quartile (p should be 0.25 for Q1, 0.75 for Q3)
pub fn quartile(sorted_values: &[f64], p: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    
    let len = sorted_values.len();
    let index = p * (len - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    
    if lower == upper {
        Some(sorted_values[lower])
    } else {
        let weight = index - lower as f64;
        Some(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)
    }
}

/// Calculate extended statistical metrics
pub fn calculate_extended_stats(values: &[f64]) -> Option<ExtendedStats> {
    if values.len() < 2 {
        return None;
    }
    
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    
    // Sort for quartiles
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let median_val = median(&sorted)?;
    let q1 = quartile(&sorted, 0.25)?;
    let q3 = quartile(&sorted, 0.75)?;
    let iqr = q3 - q1;
    
    // Variance and standard deviation
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    
    // Coefficient of variation
    let cv = if mean.abs() > 1e-10 {
        (std_dev / mean.abs()) * 100.0
    } else {
        0.0
    };
    
    // Skewness (Fisher-Pearson coefficient)
    let m3 = values.iter()
        .map(|x| (x - mean).powi(3))
        .sum::<f64>() / n;
    let skewness = if std_dev > 1e-10 {
        m3 / std_dev.powi(3)
    } else {
        0.0
    };
    
    // Kurtosis (excess kurtosis)
    let m4 = values.iter()
        .map(|x| (x - mean).powi(4))
        .sum::<f64>() / n;
    let kurtosis = if variance > 1e-10 {
        (m4 / variance.powi(2)) - 3.0
    } else {
        0.0
    };
    
    Some(ExtendedStats {
        median: median_val,
        q1,
        q3,
        iqr,
        variance,
        std_dev,
        coefficient_of_variation: cv,
        skewness,
        kurtosis,
    })
}

/// Detect outliers using IQR method
pub fn detect_outliers_iqr(value: f64, q1: f64, q3: f64, iqr: f64, k: f64) -> OutlierDetection {
    let lower_bound = q1 - k * iqr;
    let upper_bound = q3 + k * iqr;
    
    let is_outlier = value < lower_bound || value > upper_bound;
    
    // Score: how many IQRs away from the nearest fence
    let score = if value < lower_bound {
        (lower_bound - value) / iqr
    } else if value > upper_bound {
        (value - upper_bound) / iqr
    } else {
        0.0
    };
    
    OutlierDetection {
        is_outlier,
        score,
        method: OutlierMethod::IQR,
    }
}

/// Detect outliers using Z-score method
pub fn detect_outliers_zscore(value: f64, mean: f64, std_dev: f64, threshold: f64) -> OutlierDetection {
    if std_dev < 1e-10 {
        return OutlierDetection {
            is_outlier: false,
            score: 0.0,
            method: OutlierMethod::ZScore,
        };
    }
    
    let z_score = (value - mean).abs() / std_dev;
    
    OutlierDetection {
        is_outlier: z_score > threshold,
        score: z_score,
        method: OutlierMethod::ZScore,
    }
}

/// Detect outliers using Modified Z-score (MAD-based)
pub fn detect_outliers_modified_zscore(value: f64, median: f64, mad: f64, threshold: f64) -> OutlierDetection {
    if mad < 1e-10 {
        return OutlierDetection {
            is_outlier: false,
            score: 0.0,
            method: OutlierMethod::ModifiedZScore,
        };
    }
    
    let modified_z = 0.6745 * (value - median).abs() / mad;
    
    OutlierDetection {
        is_outlier: modified_z > threshold,
        score: modified_z,
        method: OutlierMethod::ModifiedZScore,
    }
}

/// Calculate Median Absolute Deviation (MAD)
pub fn calculate_mad(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let median_val = median(&sorted)?;
    
    let mut deviations: Vec<f64> = values.iter()
        .map(|&x| (x - median_val).abs())
        .collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    median(&deviations)
}

/// Calculate normalized entropy
pub fn calculate_normalized_entropy(h: f64, n: u64) -> NormalizedEntropy {
    let n_f64 = n as f64;
    
    // H_norm = H / log2(N)
    let h_norm = if n > 1 {
        h / n_f64.log2()
    } else {
        0.0
    };
    
    // Entropy rate = H / N
    let entropy_rate = if n > 0 {
        h / n_f64
    } else {
        0.0
    };
    
    NormalizedEntropy {
        h_norm,
        entropy_rate,
    }
}

/// Fit saturation model: H(N) = H_max * (1 - exp(-N/N0))
/// Using simple nonlinear least squares approximation
pub fn fit_saturation_model(data: &[(u64, f64)]) -> Option<SaturationModel> {
    if data.len() < 3 {
        return None;
    }
    
    // Initial estimates
    let h_max_estimate = data.iter()
        .map(|(_, h)| h)
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b)) * 1.1;
    
    let n_median = {
        let mut ns: Vec<u64> = data.iter().map(|(n, _)| *n).collect();
        ns.sort_unstable();
        ns[ns.len() / 2] as f64
    };
    
    // Simple two-parameter fit using logarithmic transformation
    // For large N: H ≈ H_max - H_max * exp(-N/N0)
    // ln(H_max - H) ≈ ln(H_max) - N/N0
    
    let mut valid_points = Vec::new();
    for &(n, h) in data {
        if h < h_max_estimate && h > 0.0 {
            let residual = h_max_estimate - h;
            if residual > 0.0 {
                valid_points.push((n as f64, residual.ln()));
            }
        }
    }
    
    if valid_points.len() < 2 {
        return None;
    }
    
    // Linear regression on transformed data
    let n_points = valid_points.len() as f64;
    let sum_x = valid_points.iter().map(|(x, _)| x).sum::<f64>();
    let sum_y = valid_points.iter().map(|(_, y)| y).sum::<f64>();
    let sum_xx = valid_points.iter().map(|(x, _)| x * x).sum::<f64>();
    let sum_xy = valid_points.iter().map(|(x, y)| x * y).sum::<f64>();
    
    let slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n_points;
    
    // Extract parameters
    let n0 = if slope.abs() > 1e-10 { -1.0 / slope } else { n_median };
    let h_max = E.powf(intercept);
    
    // Calculate R²
    let mean_y = sum_y / n_points;
    let ss_tot = valid_points.iter()
        .map(|(_, y)| (y - mean_y).powi(2))
        .sum::<f64>();
    let ss_res = valid_points.iter()
        .map(|(x, y)| {
            let predicted = intercept + slope * x;
            (y - predicted).powi(2)
        })
        .sum::<f64>();
    
    let r_squared = if ss_tot > 1e-10 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };
    
    Some(SaturationModel {
        h_max: h_max.max(h_max_estimate * 0.9),
        n0: n0.max(10.0),
        r_squared,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_median() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0, 5.0]), Some(3.0));
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        assert_eq!(median(&[5.0]), Some(5.0));
        assert_eq!(median(&[]), None);
    }
    
    #[test]
    fn test_quartiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert!((quartile(&data, 0.25).unwrap() - 3.0).abs() < 0.5);
        assert!((quartile(&data, 0.75).unwrap() - 7.0).abs() < 0.5);
    }
    
    #[test]
    fn test_extended_stats() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = calculate_extended_stats(&data).unwrap();
        
        assert_eq!(stats.median, 4.5);
        assert!(stats.std_dev > 0.0);
        assert!(stats.iqr > 0.0);
    }
    
    #[test]
    fn test_outlier_detection_iqr() {
        let detection = detect_outliers_iqr(100.0, 1.0, 3.0, 2.0, 1.5);
        assert!(detection.is_outlier);
        assert!(detection.score > 0.0);
    }
    
    #[test]
    fn test_normalized_entropy() {
        let norm = calculate_normalized_entropy(4.0, 100);
        assert!(norm.h_norm > 0.0 && norm.h_norm <= 1.0);
        assert_eq!(norm.entropy_rate, 0.04);
    }
}
