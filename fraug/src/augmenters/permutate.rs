use super::base::Augmenter;
use rand::rng;
use rand::seq::SliceRandom;

/// Permutate time series
pub struct Permutate {
    /// Size of series segments
    size: usize,
}

impl Permutate {
    /// Creates new permutate augmenter
    pub fn new(size: usize) -> Self {
        Permutate { size }
    }
}

impl Augmenter for Permutate {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let mut segments = x.chunks(self.size).collect::<Vec<_>>();

        segments.shuffle(&mut rng());

        segments.iter().map(|arr| arr.to_vec()).flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse() {
        let series = vec![1.0, 2.0, 3.0, 4.0];

        let aug = Permutate::new(2);
        let series = aug.augment_one(&series);

        assert!(series == vec![3.0, 4.0, 1.0, 2.0] || series == vec![1.0, 2.0, 3.0, 4.0]);
    }
}
