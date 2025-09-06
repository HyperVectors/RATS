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
    pub window_size: usize,
    /// Number of segments in window
    pub segment_size: usize,
    p: f64,
}

impl Permutate {
    /// Creates new permutate augmenter
    pub fn new(window_size: usize, segment_size: usize) -> Self {
        Permutate {
            name: "Permutate".to_string(),
            window_size,
            segment_size,
            p: 1.0,
        }
    }
}

impl Augmenter for Permutate {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let span = info_span!("", step = "augment_one");
        let _enter = span.enter();
        let mut windows = x.chunks(self.window_size).collect::<Vec<_>>();
        let mut res = Vec::with_capacity(windows.len());
        
        for window in &mut windows {
            let mut segments = window.chunks(self.segment_size).collect::<Vec<_>>();
            
            segments.shuffle(&mut rng());
            res.push(segments.iter().map(|arr| arr.to_vec()).flatten().collect::<Vec<f64>>());
        }
        
        res.iter().map(|arr| arr.to_vec()).flatten().collect()
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
