//! # RATS
//! RATS is a *f*ast *r*ust-based time series *aug*mentation library.
//! 
//! The crate provides many augmenters that work on labeled univariate time series data.
//! These can be found in the `augmenters` module. The main struct for containing the data
//! and passing it around is `Dataset`.
//! 
//! Python bindings for this crate exist as well under `RATSpy`.

pub mod augmenters;
pub mod transforms;
pub mod quality_benchmarking;

/// Holds multiple univariate time series with their labels
/// 
/// Passed to the `augment_batch` function from augmenters
pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<String>,
}
