use crate::Dataset;
use rand::prelude::*;
use rand::rng;
use rayon::prelude::*;
use std::ops::Add;

/// Trait for all augmenters, allows for augmentation of one time series or a batch
pub trait Augmenter {
    /// Augment a whole batch
    /// 
    /// Parallelized using rayon when `parallell` is set
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
    
    /// Augment one time series
    /// 
    /// When called, the augmenter will always augment the series no matter what the probability for this augmenter is
    fn augment_one(&self, x: &[f64]) -> Vec<f64>;

    /// Get the probability that this augmenter will augment a series in a batch
    fn get_probability(&self) -> f64;

    /// By setting a probability with this function the augmenter will only augment a series in a
    /// batch with the specified probability
    fn set_probability(&mut self, probability: f64);
}

/// A pipeline of augmenters
/// 
/// Executes many augmenters at once
/// 
/// # Example
/// 
/// ```
///  use fraug::Dataset;
///  use fraug::augmenters::*;
/// 
///  let series = vec![1.0; 100];
///  let mut set = Dataset {
///     features: vec![series],
///     labels: vec![String::from("1")],
///  };
/// 
///  let pipeline = AugmentationPipeline::new() 
///                 + Repeat::new(5) 
///                 + Crop::new(20)
///                 + Jittering::new(0.2);
/// 
///  pipeline.augment_batch(&mut set, true);
///
///  assert_eq!(set.features.len(), 5);
///  assert_eq!(set.features[3].len(), 20);
/// ```
pub struct AugmentationPipeline {
    pub name: String,
    augmenters: Vec<Box<dyn Augmenter + Sync>>,
    p: f64,
}

impl AugmentationPipeline {
    /// Creates an empty pipeline
    pub fn new() -> Self {
        AugmentationPipeline {
            name: "AugmentationPipeline".to_string(),
            augmenters: Vec::new(),
            p: 1.0,
        }
    }

    /// Add an augmenter to the pipeline
    /// 
    /// Has the same effect as using the `+` operator
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
