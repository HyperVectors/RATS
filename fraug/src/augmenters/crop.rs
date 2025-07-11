use super::base::Augmenter;
use rayon::prelude::*;

/// Augmenter that crops each row into a random continuous slice of specified size
/// Also known as window slicing
pub struct Crop {
    pub name: String,
    size: usize,
}

impl Crop {
    pub fn new(size: usize) -> Self {
        Crop {
            name: "Crop".to_string(),
            size,
        }
    }

    fn get_slice(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();

        if self.size >= n {
            return x.to_vec();
        }

        let start: usize = rand::random_range(0..(n - self.size + 1));

        x[start..(start + self.size)].to_vec()
    }
}

impl Augmenter for Crop {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        self.get_slice(x)
    }
}
