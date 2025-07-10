use super::base::Augmenter;

/// Augmenter that scales a time series with a random scalar within the specified range
pub struct Scaling {
    min_factor: f64,
    max_factor: f64,
}

impl Scaling {
    pub fn new(min: f64, max: f64) -> Self {
        Scaling {
            min_factor: min,
            max_factor: max,
        }
    }
}

impl Augmenter for Scaling {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let scalar = rand::random_range(self.min_factor..self.max_factor);
        x.iter().map(|val| *val * scalar).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling() {
        let series = vec![1.0; 100];

        let augmenter = Scaling::new(2.0, 4.0);
        let series = augmenter.augment_one(&series);

        series
            .iter()
            .for_each(|&val| assert!(val >= 2.0 && val <= 4.0));
    }
}
