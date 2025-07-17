//! Module to perform quality benchmarking of augmenters
//! This module provides functionality to evaluate and compare the quality of different data augmentation techniques.
//! Currently it includes using the Dynamic Time Warping (DTW) algorithm to measure the similarity between original and augmented time series data.
//! # Examples
//! ```
//! use fraug::quality_benchmarking::dtw::DynamicTimeWarping;
//! use fraug::augmenters::Jittering;
//! use fraug::Dataset;
//! let original_series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let original_reference = original_series.clone();
//! let augmenter = Jittering::new(0.1);
//! augmenter.augment_batch(&[&original_series], false, false);
//! 
//! let distance, path = dtw(&original_series, &original_reference);
//! ```

mod dtw;
pub use dtw::*;