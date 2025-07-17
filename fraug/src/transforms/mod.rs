//! Module containing various frequency domain transforms for time series data.
//! This module provides implementations of different frequency domain transforms such as Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT).
//! These transforms can be used for various purposes, including feature extraction, noise reduction, and data
//! compression in time series analysis.
//! # Examples
//! ```
//! use fraug::transforms::{FastFourier, DCT};
//! use fraug::Dataset;
//! let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let dataset = Dataset {
//!    features: vec![series],
//!   labels: vec![String::from("1")],
//! };
//! let fft = FastFourier::new();
//! let dct = DCT::new();
//! let transformed_fft = fft.transform(&dataset);
//! let transformed_dct = dct.transform(&dataset);
//! 
//! let inverse_fft = fft.inverse(&transformed_fft);
//! let inverse_dct = dct.inverse(&transformed_dct);
//! ```
pub mod fastfourier;
pub mod dct;
pub mod accuracy;
