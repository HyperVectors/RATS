use super::base::Augmenter;

/// Augmenter that drops data points in series
///
/// Drops `percentage` % of data points and replaces them with `default`
pub struct Drop {
    pub name: String,
    pub percentage: f64,
    pub default: f64,
    p: f64,
}

impl Drop {
    /// Creates new drop augmenter
    ///
    /// When `default` is `None`, it is set to `0.0`
    pub fn new(percentage: f64, default: Option<f64>) -> Self {
        Drop {
            name: "Drop".to_string(),
            percentage,
            default: default.unwrap_or(0.0),
            p: 1.0,
        }
    }
}

impl Augmenter for Drop {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|val| {
                if rand::random::<f64>() < self.percentage {
                    self.default
                } else {
                    *val
                }
            })
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
