use super::base::Augmenter;
use rand::rng;
use rand::seq::SliceRandom;

/// Permutate time series
pub struct Permutate {
    pub name: String,
    /// Size of series segments
    size: usize,
}

impl Permutate {
    /// Creates new permutate augmenter
    pub fn new(size: usize) -> Self {
        Permutate {
            name: "Permutate".to_string(),
            size,
        }
    }
}

impl Augmenter for Permutate {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let mut segments = x.chunks(self.size).collect::<Vec<_>>();

        segments.shuffle(&mut rng());

        segments.iter().map(|arr| arr.to_vec()).flatten().collect()
    }
}
