use rats::quality_benchmarking::dtw;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Class to perform quality benchmarking of augmenters
/// 
/// This module provides functionality to evaluate and compare the quality of different data augmentation techniques.
/// 
/// Currently, it includes using the Dynamic Time Warping (DTW) algorithm to measure the similarity between original and augmented time series data.
#[gen_stub_pyclass]
#[pyclass]
pub struct QualityBenchmarking;

#[gen_stub_pymethods]
#[pymethods]
impl QualityBenchmarking {
    /// Implementation of Dynamic Time Warping (DTW) algorithm.
    /// 
    /// This function computes the DTW distance between two sequences and returns the distance
    /// along with the optimal path.
    /// 
    /// # Arguments
    /// 
    /// * `a` - First sequence as a list[float].
    /// 
    /// * `b` - Second sequence as a list[float].
    /// 
    /// # Returns
    /// 
    /// A tuple containing the DTW distance (float) and a list of tuples representing the
    /// optimal path as pairs of indices (int, int).
    #[staticmethod]
    pub fn compute_dtw(a: Vec<f64>, b: Vec<f64>) -> (f64, Vec<(usize, usize)>) {
        dtw(&a, &b)
    }
}

