use std::ops::{Add, Rem};
use rand::prelude::*;

use crate::Dataset;

/// Trait for all augmenters, allows for augmentation of one time series or a whole dataset
pub trait Augmenter {
    fn augment_dataset(&self, input: &mut Dataset) {
        input.features.iter_mut().for_each(|x| self.augment_one(x));
    }

    fn augment_one(&self, x: &mut [f64]);
}

/// Augmenter that executes another augmenter on a row with a given probability p
/// 
/// Only works for augmenters that can operate on a single time series (that implement augment_one)
pub struct ConditionalAugmenter {
    inner: Box<dyn Augmenter>,
    p: f64,
}

impl ConditionalAugmenter {
    pub fn new<T: Augmenter + 'static>(augmenter: T, probabilty: f64) -> Self {
        ConditionalAugmenter { inner: Box::new(augmenter), p: probabilty}
    }
}

impl Augmenter for ConditionalAugmenter {
    fn augment_dataset(&self, input: &mut Dataset) {
        let mut rng = rand::rng();
        input.features.iter_mut().for_each(|x| if rng.random::<f64>() < self.p { self.inner.augment_one(x) });
    }

    fn augment_one(&self, x: &mut [f64]) {
        if rand::random::<f64>() < self.p {
            self.inner.augment_one(x);
        }
    }
}

/// Augmenter that includes multiple other augmenters to build a pipeline
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