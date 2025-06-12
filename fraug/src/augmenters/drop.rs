use super::base::Augmenter;

/// Augmenter that drops data points in series
/// 
/// Drops `percentage` % of data points and replaces them with `default`
pub struct Drop {
    percentage: f64,
    default: f64
}

impl Drop {
    /// Creates new drop augmenter
    /// 
    /// When `default` is `None`, it is set to `0.0`
    pub fn new(percentage: f64, default: Option<f64>) -> Self {
        Drop { percentage, default: default.unwrap_or(0.0) }
    }
}

impl Augmenter for Drop {
    fn augment_one(&self, x: &mut [f64]) {
        x.iter_mut().for_each(|val| if rand::random::<f64>() < self.percentage { *val = self.default });
    }
}
