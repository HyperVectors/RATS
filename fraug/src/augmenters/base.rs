use rand::prelude::*;
use rayon::prelude::*;
use std::ops::Add;

use crate::Dataset;

/// Trait for all augmenters, allows for augmentation of one time series or a batch
pub trait Augmenter {
    fn augment_batch(&self, input: &mut Dataset, parallel: bool)
    where
        Self: Sync,
    {
        if parallel {
            input
                .features
                .par_iter_mut()
                .for_each(|x| *x = self.augment_one(x));
        } else {
            input
                .features
                .iter_mut()
                .for_each(|x| *x = self.augment_one(x));
        }
    }

    fn augment_one(&self, x: &[f64]) -> Vec<f64>;
}

/// Augmenter that executes another augmenter on a row with a given probability p
///
/// Only works for augmenters that can operate on a single time series (that implement augment_one)
pub struct ConditionalAugmenter {
    pub name: String,
    inner: Box<dyn Augmenter>,
    p: f64,
}

impl ConditionalAugmenter {
    pub fn new<T: Augmenter + 'static>(augmenter: T, probability: f64) -> Self {
        ConditionalAugmenter {
            name: "ConditionalAugmenter".to_string(),
            inner: Box::new(augmenter),
            p: probability,
        }
    }
}

unsafe impl Sync for ConditionalAugmenter {}

impl Augmenter for ConditionalAugmenter {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let mut rng = rand::rng();
        if rng.random::<f64>() < self.p {
            self.inner.augment_one(x)
        } else {
            x.to_vec()
        }
    }
}

/// Augmenter that includes multiple other augmenters to build a pipeline
pub struct AugmentationPipeline {
    pub name: String,
    augmenters: Vec<Box<dyn Augmenter + Sync>>,
}

impl AugmentationPipeline {
    pub fn new() -> Self {
        AugmentationPipeline {
            name: "AugmentationPipeline".to_string(),
            augmenters: Vec::new(),
        }
    }

    pub fn add<T: Augmenter + 'static + Sync>(&mut self, augmenter: T) {
        self.augmenters.push(Box::new(augmenter));
    }
}

impl Augmenter for AugmentationPipeline {
    fn augment_batch(&self, input: &mut Dataset, parallel: bool) {
        self.augmenters
            .iter()
            .for_each(|augmenter| augmenter.augment_batch(input, parallel));
    }

    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let mut res = x.to_vec();
        for augmenter in self.augmenters.iter() {
            res = augmenter.augment_one(&res);
        }
        res
    }
}

impl<T: Augmenter + 'static + Sync> Add<T> for AugmentationPipeline {
    type Output = AugmentationPipeline;

    fn add(self, rhs: T) -> Self::Output {
        let mut augmenters = self.augmenters;
        augmenters.push(Box::new(rhs));

        AugmentationPipeline {
            name: "ConditionalAugmenter".to_string(),
            augmenters,
        }
    }
}
