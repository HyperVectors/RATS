//! Module containing various frequency domain transforms for time series data.
//! This module provides implementations of different frequency domain transforms such as Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT).
//! These transforms can be used for various purposes, including feature extraction, noise reduction, and data
//! compression in time series analysis.
//! # Examples
//! ```
//! use rats::transforms::fastfourier::*;
//! use rats::transforms::dct::*;
//! use rats::Dataset;
//! let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let dataset = Dataset {
//!    features: vec![series],
//!   labels: vec![String::from("1")],
//! };
//! let transformed_fft = dataset_fft(&dataset, true);
//! let transformed_dct = dataset_dct(&dataset, true);
//! 
//! let inverse_fft = dataset_ifft(&transformed_fft, true);
//! let inverse_dct = dataset_idct(&transformed_dct, true);
//! ```
pub mod fastfourier;
pub mod dct;
pub mod accuracy;
