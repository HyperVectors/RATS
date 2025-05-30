use crate::Dataset;

pub trait Augmenter {
    fn augment_dataset(&self, input: &mut Dataset) {
        input.features.iter_mut().for_each(|x| self.augment_one(x));
    }

    fn augment_one(&self, x: &mut [f64]);
}
