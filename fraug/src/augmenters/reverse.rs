use super::base::Augmenter;

/// Reverses time series
pub struct Reverse {
    pub name: String,
}

impl Reverse {
    /// Creates new reverse augmenter
    pub fn new() -> Self {
        Reverse {
            name: "Reverse".to_string(),
        }
    }
}

impl Augmenter for Reverse {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        x.iter().rev().map(|v| *v).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse() {
        let series = vec![1.0, 2.0, 3.0, 4.0];

        let aug = Reverse::new();
        let series = aug.augment_one(&series);

        assert_eq!(series, vec![4.0, 3.0, 2.0, 1.0]);
    }
}
