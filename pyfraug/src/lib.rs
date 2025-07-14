mod augmenters;
mod transforms;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, PyStubType, TypeInfo};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Holds multiple univariate time series with their labels
///
/// Passed to the `augment_batch` function from augmenters
#[gen_stub_pyclass]
#[pyclass]
pub struct Dataset {
    pub(crate) inner: fraug::Dataset,
}

impl PyStubType for &mut Dataset {
    fn type_output() -> TypeInfo {
        TypeInfo::with_module("pyfraug.Dataset", "pyfraug".into())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Dataset {
    #[new]
    fn new<'py>(features: &Bound<'py, PyArray2<f64>>, labels: Vec<String>) -> Self {
        let features: PyReadonlyArray2<f64> = features.extract().unwrap();
        let features: Array2<f64> = features.as_array().to_owned().into();

        let features: Vec<Vec<f64>> = features.rows().into_iter().map(|x| x.to_vec()).collect();

        Dataset {
            inner: fraug::Dataset { features, labels },
        }
    }

    #[getter]
    fn get_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let features = self.inner.features.clone();
        let features = Array2::from_shape_vec(
            (features.len(), features[0].len()),
            features.iter().flatten().map(|&x| x).collect(),
        )
        .unwrap();
        features.into_pyarray(py)
    }

    #[setter]
    fn set_features<'py>(&mut self, features: PyReadonlyArray2<f64>) {
        let features: Array2<f64> = features.as_array().to_owned().into();
        let features: Vec<Vec<f64>> = features.rows().into_iter().map(|x| x.to_vec()).collect();

        self.inner.features = features;
    }

    #[getter]
    fn get_labels(&self) -> Vec<String> {
        self.inner.labels.clone()
    }

    #[setter]
    fn set_labels(&mut self, labels: Vec<String>) {
        self.inner.labels = labels;
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyfraug(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Dataset>()?;
    m.add_class::<augmenters::Repeat>()?;
    m.add_class::<augmenters::Scaling>()?;
    m.add_class::<augmenters::Drop>()?;
    m.add_class::<augmenters::Crop>()?;
    m.add_class::<augmenters::Jittering>()?;
    m.add_class::<augmenters::Rotation>()?;
    m.add_class::<augmenters::NoiseType>()?;
    m.add_class::<augmenters::AddNoise>()?;
    m.add_class::<augmenters::AmplitudePhasePerturbation>()?;
    m.add_class::<augmenters::FrequencyMask>()?;
    m.add_class::<augmenters::RandomTimeWarpAugmenter>()?;
    m.add_class::<augmenters::PoolingMethod>()?;
    m.add_class::<augmenters::Pool>()?;
    m.add_class::<augmenters::Quantize>()?;
    m.add_class::<augmenters::Resize>()?;
    m.add_class::<augmenters::Reverse>()?;
    m.add_class::<augmenters::Permutate>()?;
    m.add_class::<transforms::Transforms>()?;
    m.add_class::<augmenters::Drift>()?;
    m.add_class::<augmenters::Convolve>()?;
    m.add_class::<augmenters::ConvolveWindow>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
