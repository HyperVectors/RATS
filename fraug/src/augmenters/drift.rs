use super::base::Augmenter;

/// Drifts the value of a time series
///
///
pub struct Drift {
    pub name: String,
}

impl Drift {
    /// Creates new drift augmenter
    pub fn new() -> Self {
        Drift {
            name: "Drift".to_string(),
        }
    }
}

impl Augmenter for Drift {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        todo!();
    }
}
