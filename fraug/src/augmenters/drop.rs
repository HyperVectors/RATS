use super::base::Augmenter;

/// Augmenter that drops data points in series
///
/// Drops `percentage` % of data points and replaces them with `default`
pub struct Drop {
    pub name: String,
    percentage: f64,
    default: f64,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_all() {
        let series = vec![1.0; 100];

        let drop = Drop::new(1.0, None);
        let series = drop.augment_one(&series);

        assert_eq!(series, vec![0.0; 100]);
    }

    #[test]
    fn drop_none() {
        let series = vec![1.0; 100];

        let drop = Drop::new(0.0, None);
        let series = drop.augment_one(&series);

        assert_eq!(series, vec![1.0; 100]);
    }
}
