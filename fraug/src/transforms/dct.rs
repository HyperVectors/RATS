use crate::Dataset;
use rayon::prelude::*;
use rustdct::{DctPlanner};

/// Computes Discrete Cosine Transform (DCT‐II) over each series in the dataset.
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

/// Compute the inverse Discrete Cosine Transform (DCT‐III) over each series in the dataset.
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