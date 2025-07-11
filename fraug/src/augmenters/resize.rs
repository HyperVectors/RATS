use super::base::Augmenter;

/// Changes temporal resolution of time series by changing the length
///
/// Does not interpolate values!
pub struct Resize {
    pub name: String,
    /// size after the augmentation
    size: usize,
}

impl Resize {
    /// Creates new resize augmenter
    pub fn new(size: usize) -> Self {
        Resize {
            name: "Resize".to_string(),
            size,
        }
    }
}

impl Augmenter for Resize {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let ratio = x.len() as f64 / self.size as f64;
        (0..self.size)
            .map(|i| x[(i as f64 * ratio) as usize])
            .collect()
    }
}
