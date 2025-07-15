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
    pub fn dct(dataset: &Dataset, parallel: bool) -> Dataset {
        let result = fraug::transforms::dct::dataset_dct(&dataset.inner, parallel);
        Dataset { inner: result }
    }

    #[staticmethod]
    pub fn idct(dataset: &Dataset, parallel: bool) -> Dataset {
        let result = fraug::transforms::dct::dataset_idct(&dataset.inner, parallel);
        Dataset { inner: result }
    }

    #[staticmethod]
    pub fn compare_within_tolerance(
        original: &Dataset,
        reconstructed: &Dataset,
        tolerance: f64,
    ) -> (f64, bool) {
        fraug::transforms::accuracy::compare_datasets_within_tolerance(
            &original.inner,
            &reconstructed.inner,
            tolerance,
        )
    }
}
