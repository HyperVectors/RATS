//! Module for the `Augmenter` trait and all augmenters this crate provides
//! 
//! Every augmenter implements the `Augmenter` trait which enables a unified interface
//! with the methods `augment_batch` and `augment_one`
//!
//! # Examples
//! Every augmenter can be used analogously to these examples
//! 
//! ```
//! use fraug::augmenters::{Augmenter, Rotation};
//! 
//! let series = vec![1.0; 100];
//!
//! let augmenter = Rotation::new(0.5);
//! let series = augmenter.augment_one(&series);
//!
//! assert_eq!(series, vec![0.0; 100]);
//! ```
//! 
//! ```
//! use fraug::Dataset;
//! use fraug::augmenters::*;
//!
//! let series = vec![1.0; 100];
//! let mut set = Dataset {
//!    features: vec![series],
//!    labels: vec![String::from("1")],
//! };
//! 
//! let mut jittering = Jittering::new(0.2);
//! jittering.set_probability(0.5); // Only do jittering for half of the series in the batch
//! 
//! let pipeline = AugmentationPipeline::new() 
//!                + Repeat::new(5) 
//!                + Crop::new(20)
//!                + jittering;
//!
//! pipeline.augment_batch(&mut set, true, false);
//!
//! assert_eq!(set.features.len(), 5);
//! assert_eq!(set.features[3].len(), 20);
//! ```

mod addnoise;
mod amplitude_phase_perturbation;
mod base;
mod convolve;
mod crop;
mod drift;
mod drop;
mod frequency_mask;
mod jittering;
mod permutate;
mod pool;
mod quantize;
mod repeat;
mod resize;
mod reverse;
mod rotation;
mod scaling;
mod time_warp;

pub use addnoise::{AddNoise, NoiseType};
pub use amplitude_phase_perturbation::AmplitudePhasePerturbation;
pub use base::{AugmentationPipeline, Augmenter};
pub use convolve::{Convolve, ConvolveWindow};
pub use crop::Crop;
pub use drift::Drift;
pub use drop::Drop;
pub use frequency_mask::FrequencyMask;
pub use jittering::Jittering;
pub use permutate::Permutate;
pub use pool::{Pool, PoolingMethod};
pub use quantize::Quantize;
pub use repeat::Repeat;
pub use resize::Resize;
pub use reverse::Reverse;
pub use rotation::Rotation;
pub use scaling::Scaling;
pub use time_warp::RandomTimeWarpAugmenter;
