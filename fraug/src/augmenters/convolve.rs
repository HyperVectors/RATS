use super::base::Augmenter;

/// Augmenter that convolves series with a kernel window
///
/// 
pub struct Convolve {
    
}

impl Convolve {
    /// Creates new convolve augmenter
    pub fn new() -> Self {
        Convolve { }
    }
}

impl Augmenter for Convolve {
    fn augment_one(&self, x: &mut [f64]) {
        todo!();
    }
}