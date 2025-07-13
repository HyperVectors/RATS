use super::base::Augmenter;

/// Augmenter that scales a time series with a random scalar within the specified range
pub struct Scaling {
    pub name: String,
    min_factor: f64,
    max_factor: f64,
    p: f64,
}

impl Scaling {
    pub fn new(min: f64, max: f64) -> Self {
        Scaling {
            name: "Scaling".to_string(),
            min_factor: min,
            max_factor: max,
            p: 1.0,
        }
    }
}

impl Augmenter for Scaling {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let scalar = rand::random_range(self.min_factor..self.max_factor);
        x.iter().map(|val| *val * scalar).collect()
    }

    fn get_probability(&self) -> f64 {
        self.p
    }

    fn set_probability(&mut self, probability: f64) {
        self.p = probability;
    }
}
