use super::base::Augmenter;

/// Changes temporal resolution of time series by changing the length
///
/// Does not interpolate values!
pub struct Resize {
    /// size after the augmentation
    size: usize,
}

impl Resize {
    /// Creates new resize augmenter
    pub fn new(size: usize) -> Self {
        Resize { size }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_smaller() {
        let series = vec![1.0; 90]
            .iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect::<Vec<_>>();

        let aug = Resize::new(10);
        let series = aug.augment_one(&series);

        assert_eq!(
            series,
            vec![0.0, 9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0, 81.0]
        );
    }

    #[test]
    fn resize_larger() {
        let series = vec![1.0; 5]
            .iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect::<Vec<_>>();

        let aug = Resize::new(12);
        let series = aug.augment_one(&series);

        assert_eq!(
            series,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        );
    }
}
