use crate::Dataset;
use fraug::augmenters::Augmenter;
use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;

macro_rules! wrap_augmentation_functions {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            fn augment_batch(&self, dataset: &mut Dataset, parallel: bool) {
                self.inner.augment_batch(&mut dataset.inner, parallel);
            }

            fn augment_one<'py>(
                &self,
                py: Python<'py>,
                x: &Bound<'py, PyArray1<f64>>,
            ) -> Bound<'py, PyArray1<f64>> {
                let x = x.to_owned_array();
                let x_vec = x.as_slice().unwrap().to_vec();

                let x_vec = self.inner.augment_one(&x_vec);

                let x = ndarray::Array::from_vec(x_vec);
                x.to_pyarray(py)
            }

            #[getter]
            fn name(&self) -> PyResult<String> {
                Ok(self.inner.name.clone())
            }

            #[getter]
            fn get_probability(&self) -> PyResult<f64> {
                Ok(self.inner.get_probability())
            }

            #[setter]
            fn set_probability(&mut self, p: f64) -> PyResult<()> {
                self.inner.set_probability(p);
                Ok(())
            }
        }
    };
}

#[pyclass]
pub struct Repeat {
    inner: fraug::augmenters::Repeat,
}

#[pymethods]
impl Repeat {
    #[new]
    fn new(times: usize) -> Self {
        Repeat {
            inner: fraug::augmenters::Repeat::new(times),
        }
    }
}

wrap_augmentation_functions!(Repeat);

#[pyclass]
pub struct Scaling {
    inner: fraug::augmenters::Scaling,
}

#[pymethods]
impl Scaling {
    #[new]
    fn new(min: f64, max: f64) -> Self {
        Scaling {
            inner: fraug::augmenters::Scaling::new(min, max),
        }
    }
}

wrap_augmentation_functions!(Scaling);

#[pyclass]
pub struct Rotation {
    inner: fraug::augmenters::Rotation,
}

#[pymethods]
impl Rotation {
    #[new]
    fn new(anchor: f64) -> Self {
        Rotation {
            inner: fraug::augmenters::Rotation::new(anchor),
        }
    }
}

wrap_augmentation_functions!(Rotation);

#[pyclass]
pub struct Jittering {
    inner: fraug::augmenters::Jittering,
}

#[pymethods]
impl Jittering {
    #[new]
    fn new(standard_deviation: f64) -> Self {
        Jittering {
            inner: fraug::augmenters::Jittering::new(standard_deviation),
        }
    }
}

wrap_augmentation_functions!(Jittering);

#[pyclass]
pub struct Drop {
    inner: fraug::augmenters::Drop,
}

#[pymethods]
impl Drop {
    #[new]
    #[pyo3(signature = (percentage, *, default=None))]
    fn new(percentage: f64, default: Option<f64>) -> Self {
        Drop {
            inner: fraug::augmenters::Drop::new(percentage, default),
        }
    }
}

wrap_augmentation_functions!(Drop);

#[pyclass]
pub struct Crop {
    inner: fraug::augmenters::Crop,
}

#[pymethods]
impl Crop {
    #[new]
    fn new(size: usize) -> Self {
        Crop {
            inner: fraug::augmenters::Crop::new(size),
        }
    }
}

wrap_augmentation_functions!(Crop);

#[pyclass]
pub enum NoiseType {
    Uniform,
    Gaussian,
    Spike,
    Slope,
}

#[pyclass]
pub struct AddNoise {
    inner: fraug::augmenters::AddNoise,
}

#[pymethods]
impl AddNoise {
    #[new]
    #[pyo3(signature = (noise_type, *, bounds=None, mean=None, std_dev=None))]
    fn new(
        noise_type: &NoiseType,
        bounds: Option<(f64, f64)>,
        mean: Option<f64>,
        std_dev: Option<f64>,
    ) -> Self {
        let int_noise_type = match noise_type {
            NoiseType::Uniform => fraug::augmenters::NoiseType::Uniform,
            NoiseType::Gaussian => fraug::augmenters::NoiseType::Gaussian,
            NoiseType::Spike => fraug::augmenters::NoiseType::Spike,
            NoiseType::Slope => fraug::augmenters::NoiseType::Slope,
        };

        AddNoise {
            inner: fraug::augmenters::AddNoise::new(int_noise_type, bounds, mean, std_dev),
        }
    }
}

wrap_augmentation_functions!(AddNoise);

#[pyclass]
pub struct AmplitudePhasePerturbation {
    inner: fraug::augmenters::AmplitudePhasePerturbation,
}

#[pymethods]
impl AmplitudePhasePerturbation {
    #[new]
    fn new(magnitude_std: f64, phase_std: f64, is_time_domain: bool) -> Self {
        AmplitudePhasePerturbation {
            inner: fraug::augmenters::AmplitudePhasePerturbation::new(
                magnitude_std,
                phase_std,
                is_time_domain,
            ),
        }
    }
}

wrap_augmentation_functions!(AmplitudePhasePerturbation);

#[pyclass]
pub struct FrequencyMask {
    inner: fraug::augmenters::FrequencyMask,
}

#[pymethods]
impl FrequencyMask {
    #[new]
    fn new(mask_width: usize, is_time_domain: bool) -> Self {
        FrequencyMask {
            inner: fraug::augmenters::FrequencyMask::new(mask_width, is_time_domain),
        }
    }
}

wrap_augmentation_functions!(FrequencyMask);

#[pyclass]
pub struct RandomTimeWarpAugmenter {
    inner: fraug::augmenters::RandomTimeWarpAugmenter,
}

#[pymethods]
impl RandomTimeWarpAugmenter {
    #[new]
    fn new(window_size: usize, speed_ratio_range: (f64, f64)) -> Self {
        RandomTimeWarpAugmenter {
            inner: fraug::augmenters::RandomTimeWarpAugmenter::new(window_size, speed_ratio_range),
        }
    }
}

wrap_augmentation_functions!(RandomTimeWarpAugmenter);

#[pyclass]
pub enum PoolingMethod {
    Max,
    Min,
    Average,
}

#[pyclass]
pub struct Pool {
    inner: fraug::augmenters::Pool,
}

#[pymethods]
impl Pool {
    #[new]
    fn new(kind: &PoolingMethod, size: usize) -> Self {
        let int_kind = match kind {
            PoolingMethod::Max => fraug::augmenters::PoolingMethod::Max,
            PoolingMethod::Min => fraug::augmenters::PoolingMethod::Min,
            PoolingMethod::Average => fraug::augmenters::PoolingMethod::Average,
        };

        Pool {
            inner: fraug::augmenters::Pool::new(int_kind, size),
        }
    }
}

wrap_augmentation_functions!(Pool);

#[pyclass]
pub struct Quantize {
    inner: fraug::augmenters::Quantize,
}

#[pymethods]
impl Quantize {
    #[new]
    fn new(levels: usize) -> Self {
        Quantize {
            inner: fraug::augmenters::Quantize::new(levels),
        }
    }
}

wrap_augmentation_functions!(Quantize);

#[pyclass]
pub struct Resize {
    inner: fraug::augmenters::Resize,
}

#[pymethods]
impl Resize {
    #[new]
    fn new(size: usize) -> Self {
        Resize {
            inner: fraug::augmenters::Resize::new(size),
        }
    }
}

wrap_augmentation_functions!(Resize);

#[pyclass]
pub struct Reverse {
    inner: fraug::augmenters::Reverse,
}

#[pymethods]
impl Reverse {
    #[new]
    fn new() -> Self {
        Reverse {
            inner: fraug::augmenters::Reverse::new(),
        }
    }
}

wrap_augmentation_functions!(Reverse);

#[pyclass]
pub struct Permutate {
    inner: fraug::augmenters::Permutate,
}

#[pymethods]
impl Permutate {
    #[new]
    fn new(size: usize) -> Self {
        Permutate {
            inner: fraug::augmenters::Permutate::new(size),
        }
    }
}

wrap_augmentation_functions!(Permutate);

#[pyclass]
pub struct Drift {
    inner: fraug::augmenters::Drift,
}

#[pymethods]
impl Drift {
    #[new]
    fn new(max_drift: f64, n_drift_points: usize) -> Self {
        Drift {
            inner: fraug::augmenters::Drift::new(max_drift, n_drift_points),
        }
    }
}

wrap_augmentation_functions!(Drift);

#[pyclass]
pub enum ConvolveWindow {
    Flat,
    Gaussian,
}

#[pyclass]
pub struct Convolve {
    inner: fraug::augmenters::Convolve,
}

#[pymethods]
impl Convolve {
    #[new]
    fn new(window: &ConvolveWindow, size: usize) -> Self {
        let int_window = match window {
            ConvolveWindow::Flat => fraug::augmenters::ConvolveWindow::Flat,
            ConvolveWindow::Gaussian => fraug::augmenters::ConvolveWindow::Gaussian,
        };
        Convolve {
            inner: fraug::augmenters::Convolve::new(int_window, size),
        }
    }
}

wrap_augmentation_functions!(Convolve);
