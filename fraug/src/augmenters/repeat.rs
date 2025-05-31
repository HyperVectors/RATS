use crate::Dataset;

use super::base::Augmenter;

/// Augmenter that repeats all data rows n times
pub struct Repeat {
    n: usize,
}

impl Repeat {
    pub fn new(times: usize) -> Self {
        Repeat { n: times }
    }
}

impl Augmenter for Repeat {
    fn augment_dataset(&self, input: &mut Dataset) {
        let mut new_features: Vec<Vec<f64>> = Vec::with_capacity(input.features.len() * self.n);
        let mut new_labels: Vec<String> = Vec::with_capacity(input.labels.len() * self.n);

        input.features.iter().enumerate().for_each(|(i, x)| {
            for _ in 0..self.n {
                new_features.push(x.clone());
                new_labels.push(input.labels[i].clone());
            }
        });

        input.features = new_features;
        input.labels = new_labels;
    }

    fn augment_one(&self, _x: &mut [f64]) {}
}
