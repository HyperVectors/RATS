use super::base::Augmenter;
use tracing::{info_span};
/// Quantize time series to a level set
///
/// The level set is constructed by uniformly discretizing the range of all values in the series
pub struct Quantize {
    pub name: String,
    /// Number of levels in the level set
    levels: usize,
    p: f64,
}

impl Quantize {
    /// Creates new quantize augmenter
    pub fn new(levels: usize) -> Self {
        Quantize {
            name: "Quantize".to_string(),
            levels,
            p: 1.0,
        }
    }
}

impl Augmenter for Quantize {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let span = info_span!("", step = "augment_one");
        let _enter = span.enter();
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
