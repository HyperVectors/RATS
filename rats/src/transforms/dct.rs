use crate::Dataset;
use rayon::prelude::*;
use rustdct::{DctPlanner};

/// Discrete Cosine Transform (DCT-II) for time series data.
/// Converts each real-valued time series in the dataset into DCT coefficients (real, frequency representation)
pub fn dataset_dct(dataset: &Dataset, parallel: bool) -> Dataset {
    let dct_features: Vec<Vec<f64>> = if parallel {
        dataset
            .features
            .par_iter()
            .map(|sample| {
                let len = sample.len();
                let mut planner = DctPlanner::new();
                let dct = planner.plan_dct2(len);
                let mut buffer = sample.clone();
                dct.process_dct2(&mut buffer);
                buffer
            })
            .collect()
    } else {
        dataset
            .features
            .iter()
            .map(|sample| {
                let len = sample.len();
                let mut planner = DctPlanner::new();
                let dct = planner.plan_dct2(len);
                let mut buffer = sample.clone();
                dct.process_dct2(&mut buffer);
                buffer
            })
            .collect()
    };

    Dataset {
        features: dct_features,
        labels: dataset.labels.clone(),
    }
}

/// Inverse Discrete Cosine Transform (DCT-III) for time series data.
/// Reconstructs each time series from its DCT coefficients, recovering the original signal
pub fn dataset_idct(dataset: &Dataset, parallel: bool) -> Dataset {
    let time_features: Vec<Vec<f64>> = if parallel {
        dataset
            .features
            .par_iter()
            .map(|sample| {
                let len = sample.len();
                let mut planner = DctPlanner::new();
                let idct = planner.plan_dct3(len);
                let mut buffer = sample.clone();
                idct.process_dct3(&mut buffer);
                // normalize
                let norm = 2.0 / (len as f64);
                for v in &mut buffer {
                    *v *= norm;
                }
                buffer
            })
            .collect()
    } else {
        dataset
            .features
            .iter()
            .map(|sample| {
                let len = sample.len();
                let mut planner = DctPlanner::new();
                let idct = planner.plan_dct3(len);
                let mut buffer = sample.clone();
                idct.process_dct3(&mut buffer);
                let norm = 2.0 / (len as f64);
                for v in &mut buffer {
                    *v *= norm;
                }
                buffer
            })
            .collect()
    };

    Dataset {
        features: time_features,
        labels: dataset.labels.clone(),
    }
}