use crate::Dataset;

use super::base::Augmenter;

/// Augmenter that repeats all data rows n times
pub struct Repeat {
    pub name: String,
    n: usize,
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
        let features: Vec<Vec<f64>> = input.features.clone();
        let labels: Vec<String> = input.labels.clone();

        for _ in 0..self.n - 1 {
            input.features.append(&mut features.clone());
            input.labels.append(&mut labels.clone());
        }
    }

    fn augment_one(&self, _x: &[f64]) -> Vec<f64> {
        unimplemented!("Repeat augmenter only works on a dataset directly!");
    }

    fn get_probability(&self) -> f64 {
        self.p
    }

    fn set_probability(&mut self, probability: f64) {
        print!(
            "It is not possible to change the probability of {}: ",
            self.name
        );
    }
}
