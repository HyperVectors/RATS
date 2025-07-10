use super::base::Augmenter;

/// Drifts the value of a time series
///
///
pub struct Drift {

}

impl Drift {
    /// Creates new drift augmenter
    pub fn new() -> Self {
        Drift { }
    }
}

impl Augmenter for Drift {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        todo!();
    }
}