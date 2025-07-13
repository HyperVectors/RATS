use super::base::Augmenter;
use rand::rng;
use rand::seq::SliceRandom;
use tracing::{info_span};
/// Permutate time series
/// 
/// First, slices each series into segments and then rearranges them randomly
pub struct Permutate {
    pub name: String,
    /// Size of series segments
    pub size: usize,
    p: f64,
}

impl Permutate {
    /// Creates new permutate augmenter
    pub fn new(size: usize) -> Self {
        Permutate {
            name: "Permutate".to_string(),
            size,
            p: 1.0,
        }
    }
}

impl Augmenter for Permutate {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let span = info_span!("", step = "augment_one");
        let _enter = span.enter();
        let mut segments = x.chunks(self.size).collect::<Vec<_>>();

        segments.shuffle(&mut rng());

        segments.iter().map(|arr| arr.to_vec()).flatten().collect()
    }

    fn get_probability(&self) -> f64 {
        self.p
    }

    fn set_probability(&mut self, probability: f64) {
        self.p = probability;
    }

    fn get_name(&self) ->String {
        self.name.clone()
    }
}
