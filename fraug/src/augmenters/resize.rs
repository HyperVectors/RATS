use super::base::Augmenter;

/// Changes temporal resolution of time series by changing the length
///
/// Does not interpolate values!
pub struct Resize {
    pub name: String,
    /// size after the augmentation
    size: usize,
    p: f64,
}

impl Resize {
    /// Creates new resize augmenter
    pub fn new(size: usize) -> Self {
        Resize {
            name: "Resize".to_string(),
            size,
            p: 1.0,
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
