use super::base::Augmenter;

/// Quantize time series to a level set
///
/// The level set is constructed by uniformly discretizing the range of all values in the series
pub struct Quantize {
    pub name: String,
    /// Number of levels in the level set
    levels: usize,
}

impl Quantize {
    /// Creates new quantize augmenter
    pub fn new(levels: usize) -> Self {
        Quantize {
            name: "Quantize".to_string(),
            levels,
        }
    }
}

impl Augmenter for Quantize {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max - min;
        let step = range / self.levels as f64;
        let level_set = (0..self.levels)
            .map(|level| min + level as f64 * step)
            .collect::<Vec<_>>();

        // Could be faster using e.g. binary search
        x.iter()
            .map(|v| {
                let i = level_set
                    .iter()
                    .map(|&l| (l - *v).abs())
                    .enumerate()
                    .fold(
                        (0, f64::INFINITY),
                        |(i, a), (j, b)| if a > b { (j, b) } else { (i, a) },
                    )
                    .0;
                level_set[i]
            })
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize() {
        let series = vec![1.0; 11]
            .iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect::<Vec<_>>();

        let aug = Quantize::new(5);
        let series = aug.augment_one(&series);

        assert_eq!(
            series,
            vec![0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0]
        );
    }
}
