use crate::Dataset;
use pyo3::prelude::*;

#[pyclass]
pub struct Transforms;

#[pymethods]
impl Transforms {
    #[staticmethod]
    pub fn fft(dataset: &Dataset, parallel: bool) -> Dataset {
        let result = fraug::transforms::fastfourier::dataset_fft(&dataset.inner, parallel);
        Dataset { inner: result }
    }

    #[staticmethod]
    pub fn ifft(dataset: &Dataset, parallel: bool) -> Dataset {
        let result = fraug::transforms::fastfourier::dataset_ifft(&dataset.inner, parallel);
        Dataset { inner: result }
    }

    #[staticmethod]
    pub fn compare_within_tolerance(
        original: &Dataset,
        reconstructed: &Dataset,
        tolerance: f64,
    ) -> (f64, bool) {
        fraug::transforms::fastfourier::compare_datasets_within_tolerance(
            &original.inner,
            &reconstructed.inner,
            tolerance,
        )
    }
}
