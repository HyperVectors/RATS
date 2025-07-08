mod augmenters;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
struct Dataset {
    inner: fraug::Dataset,
}

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
    m.add_class::<augmenters::DynamicTimeWarpAugmenter>()?;
    m.add_class::<augmenters::FrequencyMask>()?;
    m.add_class::<augmenters::RandomWindowWarpAugmenter>()?;
    m.add_class::<augmenters::PyAugmenter>()?;
    Ok(())
}
