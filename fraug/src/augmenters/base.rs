use std::ops::Add;

use crate::Dataset;

pub trait Augmenter {
    fn augment_dataset(&self, input: &mut Dataset) {
        input.features.iter_mut().for_each(|x| self.augment_one(x));
    }

    fn augment_one(&self, x: &mut [f64]);
}

pub struct AugmentationPipeline {
    augmenters: Vec<Box<dyn Augmenter>>
}

impl AugmentationPipeline {
    pub fn new() -> Self {
        AugmentationPipeline { augmenters: Vec::new() }
    }

    pub fn add<T: Augmenter + 'static>(&mut self, augmenter: T) {
        self.augmenters.push(Box::new(augmenter));
    }
}

impl Augmenter for AugmentationPipeline {
    fn augment_dataset(&self, input: &mut Dataset){
        self.augmenters.iter().for_each(|augmenter| augmenter.augment_dataset(input));
    }

    fn augment_one(&self, x: &mut [f64]){
        self.augmenters.iter().for_each(|augmenter| augmenter.augment_one(x));
    }
}

impl<T: Augmenter + 'static> Add<T> for AugmentationPipeline {
    type Output = AugmentationPipeline;

    fn add(self, rhs: T) -> Self::Output {
        let mut augmenters = self.augmenters;
        augmenters.push(Box::new(rhs));

        AugmentationPipeline{ augmenters: augmenters }
    }
}