use super::base::Augmenter;
use rand::prelude::*;
use rand_distr::Normal;

/// Augmenter that adds white gaussian noise of the specified standard deviation and a mean of 0
pub struct Jittering {
    deviation: f64
}

impl Jittering {
    pub fn new(standard_deviation: f64) -> Self {
        Jittering { deviation: standard_deviation }
    }
}

impl Augmenter for Jittering {
    fn augment_one(&self, x: &mut [f64]) {
        let mut rng = rand::rng();
        let dist = Normal::new(0.0, self.deviation)
            .expect("Couldn't create normal distribution from specified standard deviation");
        x.iter_mut().for_each(|val| *val += dist.sample(&mut rng));
    }
}
