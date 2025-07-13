use super::base::Augmenter;
use rayon::prelude::*;

/// Augmenter that crops each series into a random continuous slice of specified `size`
/// 
/// Also known as window slicing
pub struct Crop {
    pub name: String,
    pub size: usize,
    p: f64,
}

impl Crop {
    pub fn new(size: usize) -> Self {
        Crop {
            name: "Crop".to_string(),
            size,
            p: 1.0,
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

    fn get_probability(&self) -> f64 {
        self.p
    }

    /// Not implemented!
    /// 
    /// The `Crop` augmenter always augments all rows so that all 
    /// series in a batch have the same length
    fn set_probability(&mut self, _probability: f64) {
        unimplemented!(
            "It is not possible to change the probability of {}: ",
            self.name
        );
    }

    fn get_name(&self) ->String {
        self.name.clone()
    }
}
