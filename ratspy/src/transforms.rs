use crate::Dataset;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Class containing various frequency domain transforms for time series data.
/// 
/// This module provides implementations of different frequency domain transforms such as Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT).
/// 
/// These transforms can be used for various purposes, including feature extraction, noise reduction, and data
/// compression in time series analysis.
#[gen_stub_pyclass]
#[pyclass]
pub struct Transforms;

#[gen_stub_pymethods]
#[pymethods]
impl Transforms {
    /// Converts each real-valued time series in the dataset into its frequency domain representation,
    /// storing the result as interleaved real and imaginary parts: [re0, im0, re1, im1, ...]
    #[staticmethod]
    pub fn fft(dataset: &Dataset, parallel: bool) -> Dataset {
        let result = rats::transforms::fastfourier::dataset_fft(&dataset.inner, parallel);
        Dataset { inner: result }
    }

    /// Reconstructs each time series from its frequency domain representation (interleaved real/imag parts).
    #[staticmethod]
    pub fn ifft(dataset: &Dataset, parallel: bool) -> Dataset {
        let result = rats::transforms::fastfourier::dataset_ifft(&dataset.inner, parallel);
        Dataset { inner: result }
    }

    /// Discrete Cosine Transform (DCT-II) for time series data.
    /// 
    /// Converts each real-valued time series in the dataset into DCT coefficients (real, frequency representation)
    #[staticmethod]
    pub fn dct(dataset: &Dataset, parallel: bool) -> Dataset {
        let result = rats::transforms::dct::dataset_dct(&dataset.inner, parallel);
        Dataset { inner: result }
    }

    /// Inverse Discrete Cosine Transform (DCT-III) for time series data.
    /// Reconstructs each time series from its DCT coefficients, recovering the original signal.
    #[staticmethod]
    pub fn idct(dataset: &Dataset, parallel: bool) -> Dataset {
        let result = rats::transforms::dct::dataset_idct(&dataset.inner, parallel);
        Dataset { inner: result }
    }

    /// Computes maximum absolute difference between two Datasets and check if all differences are within a tolerance.
    #[staticmethod]
    pub fn compare_within_tolerance(
        original: &Dataset,
        reconstructed: &Dataset,
        tolerance: f64,
    ) -> (f64, bool) {
        rats::transforms::accuracy::compare_datasets_within_tolerance(
            &original.inner,
            &reconstructed.inner,
            tolerance,
        )
    }
}
