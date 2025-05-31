use super::base::Augmenter;

/// Augmenter that rotates the data 180 degrees around specified anchor
pub struct Crop { 
    size: usize
}

impl Crop {
    pub fn new(size: usize) -> Self {
        Crop { size: size }
    }

    fn get_slice(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();

        if self.size >= n {
            return x.to_vec();
        }

        let start: usize = rand::random_range(0..(n-self.size));

        x[start..(start+self.size)].to_vec()
    }
}

impl Augmenter for Crop {
    fn augment_dataset(&self, input: &mut crate::Dataset) {
        let mut new_features: Vec<Vec<f64>> = Vec::with_capacity(input.features.len());

        input.features.iter().for_each(|x| new_features.push(self.get_slice(x)) );

        input.features = new_features;
    }

    fn augment_one(&self, _x: &mut [f64]) {}
}
