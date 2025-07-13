use crate::Dataset;
use rand::prelude::*;
use rand::rng;
use rayon::prelude::*;
use std::ops::Add;

/// Trait for all augmenters, allows for augmentation of one time series or a batch
pub trait Augmenter {
    fn augment_batch(&self, input: &mut Dataset, parallel: bool)
    where
        Self: Sync,
    {
        if parallel {
            input.features.par_iter_mut().for_each(|x| {
                if self.get_probability() > rng().random() {
                    *x = self.augment_one(x)
                }
            });
        } else {
            input.features.iter_mut().for_each(|x| {
                if self.get_probability() > rng().random() {
                    *x = self.augment_one(x)
                }
            });
        }
    }

    fn augment_one(&self, x: &[f64]) -> Vec<f64>;

    fn get_probability(&self) -> f64;

    /// By setting a probability with this function the augmenter will only augment a series in a
    /// batch with the specified probability
    fn set_probability(&mut self, probability: f64);
}

/// Augmenter that includes multiple other augmenters to build a pipeline
pub struct AugmentationPipeline {
    pub name: String,
    augmenters: Vec<Box<dyn Augmenter + Sync>>,
    p: f64,
}

impl AugmentationPipeline {
    pub fn new() -> Self {
        AugmentationPipeline {
            name: "AugmentationPipeline".to_string(),
            augmenters: Vec::new(),
            p: 1.0,
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

    fn get_probability(&self) -> f64 {
        self.p
    }

    fn set_probability(&mut self, probability: f64) {
        self.p = probability;
    }
}

impl<T: Augmenter + 'static + Sync> Add<T> for AugmentationPipeline {
    type Output = AugmentationPipeline;

    fn add(self, rhs: T) -> Self::Output {
        let mut augmenters = self.augmenters;
        augmenters.push(Box::new(rhs));

        AugmentationPipeline {
            name: "AugmentationPipeline".to_string(),
            augmenters,
            p: self.p,
        }
    }
}
