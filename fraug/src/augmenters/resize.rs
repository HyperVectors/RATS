use super::base::Augmenter;

/// Changes temporal resolution of time series by changing the length
pub struct Resize {
    size: usize
}

impl Resize {
    /// Creates new drift augmenter
    pub fn new(size: usize) -> Self {
        Resize { size }
    }
}

impl Augmenter for Resize {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        Vec::new()
    }
}