use crate::Dataset;

use super::base::Augmenter;
use tracing::{info_span};
/// Augmenter that repeats all data rows `n` times
/// 
/// Resource intensive because the data needs to be copied `n` times
/// 
/// Only works with `augment_batch` because the data needs to be cloned
pub struct Repeat {
    pub name: String,
    pub n: usize,
    p: f64,
}

impl Repeat {
    pub fn new(times: usize) -> Self {
        assert!(times > 0);
        Repeat {
            name: "Repeat".to_string(),
            n: times,
            p: 1.0,
        }
    }
}

impl Augmenter for Repeat {
    fn augment_batch(&self, input: &mut Dataset, _parallel: bool) {
        
        let span = info_span!("", component = self.get_name());
        let _enter = span.enter();
   
        let features: Vec<Vec<f64>> = input.features.clone();
        let labels: Vec<String> = input.labels.clone();

        for _ in 0..self.n - 1 {
            input.features.append(&mut features.clone());
            input.labels.append(&mut labels.clone());
        }
    }
    
    /// Not implemented!
    fn augment_one(&self, _x: &[f64]) -> Vec<f64> {
        let span = info_span!("", step = "augment_one");
        let _enter = span.enter();
        unimplemented!("Repeat augmenter only works on a dataset directly!");
    }

    fn get_probability(&self) -> f64 {
        self.p
    }
    
    /// Not implemented!
    fn set_probability(&mut self, _probability: f64) {
        unimplemented!(
            "It is not possible to change the probability of {}",
            self.name
        );
    }

    fn get_name(&self) ->String {
        self.name.clone()
    }
}
