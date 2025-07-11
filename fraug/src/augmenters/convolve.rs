use super::base::Augmenter;

/// Augmenter that convolves series with a kernel window
///
///
pub struct Convolve {
    pub name: String,
}

impl Convolve {
    /// Creates new convolve augmenter
    pub fn new() -> Self {
        Convolve {
            name: "Convolve".to_string(),
        }
    }
}

impl Augmenter for Convolve {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        todo!();
    }
}
