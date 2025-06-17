use super::base::Augmenter;
use crate::Dataset;

pub struct DynamicTimeWarpAugmenter;

pub fn dtw(a: &[f64], b: &[f64]) -> (f64, Vec<(usize, usize)>) {
    let n = a.len();
    let m = b.len();
    let mut cost = vec![vec![f64::INFINITY; m + 1]; n + 1];
    cost[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let diff = (a[i - 1] - b[j - 1]).abs();
            let prev = cost[i - 1][j]
                .min(cost[i][j - 1])
                .min(cost[i - 1][j - 1]);
            cost[i][j] = diff + prev;
        }
    }
    let distance = cost[n][m];

    let mut path = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        path.push((i.saturating_sub(1), j.saturating_sub(1)));
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let c_diag = cost[i - 1][j - 1];
            let c_up = cost[i - 1][j];
            let c_left = cost[i][j - 1];
            if c_diag <= c_up && c_diag <= c_left {
                i -= 1;
                j -= 1;
            } else if c_up < c_left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
    }
    path.reverse();
    (distance, path)
}

impl DynamicTimeWarpAugmenter {
    pub fn new() -> Self {
        DynamicTimeWarpAugmenter
    }

    /// Augment all samples in the dataset in-place by appending DTW-warped series.
    /// This will help us for graph plotting and checking if the warping worked later so just appending aug to original
    pub fn augment_dataset(&self, data: &mut Dataset) {
        let original_features = data.features.clone();
        let original_labels = data.labels.clone();
        let n = original_features.len();

        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let (_dist, path) = dtw(&original_features[i], &original_features[j]);
                let target_len = original_features[j].len();
                let mut accum: Vec<Vec<f64>> = vec![Vec::new(); target_len];
                for &(ai, bj) in &path {
                    accum[bj].push(original_features[i][ai]);
                }
                let warped: Vec<f64> = accum.into_iter().enumerate().map(|(idx, vals)| {
                    if !vals.is_empty() {
                        let sum: f64 = vals.iter().sum();
                        sum / vals.len() as f64
                    } else {
                        let xi = idx.min(original_features[i].len() - 1);
                        original_features[i][xi]
                    }
                }).collect();

                data.features.push(warped);
                data.labels.push(original_labels[i].clone());
            }
        }
    }
}