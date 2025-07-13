use super::base::Augmenter;
use crate::Dataset;
use rand::{Rng, rng};

pub struct DynamicTimeWarpAugmenter {
    pub name: String,
    window_size: usize,
}

pub fn dtw(a: &[f64], b: &[f64]) -> (f64, Vec<(usize, usize)>) {
    let n = a.len();
    let m = b.len();
    let mut cost = vec![vec![f64::INFINITY; m + 1]; n + 1];
    cost[0][0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let diff = (a[i - 1] - b[j - 1]).abs();
            let min_prev = cost[i - 1][j].min(cost[i][j - 1]).min(cost[i - 1][j - 1]);
            cost[i][j] = diff + min_prev;
        }
    }
    let distance = cost[n][m];
    // Backtrack to get the best path
    let mut path = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        path.push((i - 1, j - 1));
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let diag = cost[i - 1][j - 1];
            let up = cost[i - 1][j];
            let left = cost[i][j - 1];
            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up < left {
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
    pub fn new(window_size: usize) -> Self {
        DynamicTimeWarpAugmenter {
            name: "DynamicTimeWarpAugmenter".to_string(),
            window_size: window_size,
        }
    }
}

impl Augmenter for DynamicTimeWarpAugmenter {
    /// Augment all samples in the dataset in-place by appending DTW-warped series.
    // This will help us for graph plotting and checking if the warping worked later so just appending aug to original
    fn augment_batch(&self, data: &mut Dataset, _parallel: bool) {
        let originals = data.features.clone();
        let orig_labels = data.labels.clone();
        if self.window_size == 0 || self.window_size > originals.len() {}
        let mut rng = rng();

        // Slide window over series indices
        for start in 0..=(originals.len() - self.window_size) {
            // prepare uniform distribution once per window
            let i = rng.random_range(start..start + self.window_size);
            let mut j = rng.random_range(start..start + self.window_size);
            // ensure distinct
            while j == i {
                j = rng.random_range(start..start + self.window_size);
            }
            // compute DTW path
            let (_dist, path) = dtw(&originals[i], &originals[j]);
            let len = originals[i].len(); // same as originals[j].len()
            let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); len];
            for &(ai, bj) in &path {
                buckets[bj].push(originals[i][ai]);
            }
            // average or fallback directly from originals
            let warped: Vec<f64> = buckets
                .into_iter()
                .enumerate()
                .map(|(idx, vals)| {
                    if !vals.is_empty() {
                        vals.iter().copied().sum::<f64>() / vals.len() as f64
                    } else {
                        originals[i][idx]
                    }
                })
                .collect();
            data.features.push(warped);
            data.labels.push(orig_labels[i].clone());
        }
    }

    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        unimplemented!("Use augment_dataset instead!");
    }

    fn get_probability(&self) -> f64 {
        todo!()
    }

    fn set_probability(&mut self, probability: f64) {
        todo!()
    }

    fn get_name(&self) ->String {
        self.name.clone()
    }
}


