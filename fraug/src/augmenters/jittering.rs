use super::base::Augmenter;
use rand::prelude::*;
use rand_distr::Normal;

/// Augmenter that adds white gaussian noise of the specified standard deviation and a mean of 0
pub struct Jittering {
    deviation: f64,
}

impl Jittering {
    pub fn new(standard_deviation: f64) -> Self {
        Jittering {
            deviation: standard_deviation,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian() {
        let series = vec![1.0; 100];

        let augmenter = Jittering::new(0.5);
        let series = augmenter.augment_one(&series);

        assert_ne!(series, vec![1.0; 100]);
    }
}
