use crate::Dataset;

use super::base::Augmenter;

/// Augmenter that repeats all data rows n times
pub struct Repeat {
    n: usize,
}

impl Repeat {
    pub fn new(times: usize) -> Self {
        assert!(times > 0);
        Repeat { n: times }
    }
}

impl Augmenter for Repeat {
    fn augment_batch(&self, input: &mut Dataset, _parallel: bool) {
        let mut new_features: Vec<Vec<f64>> = Vec::with_capacity(input.features.len() * self.n);
        let mut new_labels: Vec<String> = Vec::with_capacity(input.labels.len() * self.n);

        input.features.iter().enumerate().for_each(|(i, x)| {
            for _ in 0..self.n {
                new_features.push(x.clone());
                new_labels.push(input.labels[i].clone());
            }
        });

        input.features = new_features;
        input.labels = new_labels;
    }

    fn augment_one(&self, _x: &[f64]) -> Vec<f64> {
        unimplemented!("Repeat augmenter only works on the dataset directly!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn repeat_2() {
        let series = vec![1.0; 100];
        let mut set = Dataset {
            features: vec![series],
            labels: vec![String::from("1")],
        };

        let augmenter = Repeat::new(2);
        augmenter.augment_batch(&mut set, false);

        assert_eq!(set.features[0], vec![1.0; 100]);
        assert_eq!(set.features[1], vec![1.0; 100]);
        assert_eq!(set.features.len(), 2);
        assert_eq!(set.labels, vec![String::from("1"); 2]);
    }
}
