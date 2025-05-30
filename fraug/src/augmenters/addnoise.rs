use super::base::Augmenter;
use rand::{distr::Uniform, prelude::*};

// Augmenter that adds noise along a simple uniform distribution with the given bounds
pub struct AddNoise {
    bounds: (f64, f64),
}

impl AddNoise {
    pub fn new(bounds: (f64, f64)) -> Self {
        AddNoise { bounds: bounds }
    }
}

impl Augmenter for AddNoise {
    fn augment_one(&self, x: &mut [f64]) {
        let mut rng = rand::rng();
        let dist = Uniform::new(self.bounds.0, self.bounds.1)
            .expect("Couldn't create uniform distribution from specified bounds");
        x.iter_mut().for_each(|val| *val += dist.sample(&mut rng));
    }
}
