use super::base::Augmenter;
use rayon::prelude::*;

/// Augmenter that crops each row into a random continuous slice of specified size
/// Also known as window slicing
pub struct Crop {
    size: usize,
}

impl Crop {
    pub fn new(size: usize) -> Self {
        Crop { size }
    }

    fn get_slice(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();

        if self.size >= n {
            return x.to_vec();
        }

        let start: usize = rand::random_range(0..(n - self.size + 1));

        x[start..(start + self.size)].to_vec()
    }
}

impl Augmenter for Crop {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        self.get_slice(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn crop_larger() {
        let series = vec![1.0; 100];
        let mut set = Dataset {
            features: vec![series],
            labels: vec![String::from("1")],
        };

        let augmenter = Crop::new(200);
        augmenter.augment_dataset(&mut set, true);

        assert_eq!(set.features[0], vec![1.0; 100]);
    }

    #[test]
    fn crop_smaller() {
        let series = vec![1.0; 100];
        let mut set = Dataset {
            features: vec![series],
            labels: vec![String::from("1")],
        };

        let augmenter = Crop::new(50);
        augmenter.augment_dataset(&mut set, true);

        assert_eq!(set.features[0], vec![1.0; 50]);
    }
}
