use super::base::Augmenter;

/// Reverses time series
pub struct Reverse {
    pub name: String,
    p: f64,
}

impl Reverse {
    /// Creates new reverse augmenter
    pub fn new() -> Self {
        Reverse {
            name: "Reverse".to_string(),
            p: 1.0,
        }
    }
}

impl Augmenter for Reverse {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        x.iter().rev().map(|v| *v).collect()
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
