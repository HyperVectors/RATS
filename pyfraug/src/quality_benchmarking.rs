use fraug::quality_benchmarking::dtw;
use pyo3::prelude::*;


#[pyclass]
pub struct QualityBenchmarking;

#[pymethods]
impl QualityBenchmarking {
    #[staticmethod]
    pub fn compute_dtw(a: Vec<f64>, b: Vec<f64>) -> (f64, Vec<(usize, usize)>) {
        dtw(&a, &b)
    }
}

