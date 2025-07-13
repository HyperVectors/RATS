use super::base::Augmenter;
use rand::prelude::*;
use rand_distr::Normal;

/// Augmenter that adds white gaussian noise of the specified standard deviation and a mean of 0
pub struct Jittering {
    pub name: String,
    deviation: f64,
    p: f64,
}

impl Jittering {
    pub fn new(standard_deviation: f64) -> Self {
        Jittering {
            name: "Jittering".to_string(),
            deviation: standard_deviation,
            p: 1.0,
        }
    }
}

impl Augmenter for Jittering {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let mut rng = rand::rng();
        let dist = Normal::new(0.0, self.deviation)
            .expect("Couldn't create normal distribution from specified standard deviation");
        x.iter().map(|val| *val + dist.sample(&mut rng)).collect()
    }

    fn get_probability(&self) -> f64 {
        self.p
    }

    fn set_probability(&mut self, probability: f64) {
        self.p = probability;
    }
}
