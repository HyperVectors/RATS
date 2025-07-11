use super::base::Augmenter;

/// Reverses time series
pub struct Reverse {
    pub name: String,
}

impl Reverse {
    /// Creates new reverse augmenter
    pub fn new() -> Self {
        Reverse {
            name: "Reverse".to_string(),
        }
    }
}

impl Augmenter for Reverse {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        x.iter().rev().map(|v| *v).collect()
    }
}
