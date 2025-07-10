use crate::Dataset;
use fraug::augmenters::Augmenter;
use numpy::{ PyArray1, PyArrayMethods, ToPyArray };
use pyo3::prelude::*;

macro_rules! wrap_augmentation_functions {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            fn augment_dataset(&self, dataset: &mut Dataset, parallel: bool) {
                self.inner.augment_dataset(&mut dataset.inner, parallel);
            }

            fn augment_one<'py>(&self, py: Python<'py>, x: &Bound<'py, PyArray1<f64>>) -> Bound<'py, PyArray1<f64>> {
                let x = x.to_owned_array();
                let x_vec = x.as_slice().unwrap().to_vec();

                let x_vec = self.inner.augment_one(&x_vec);

                let x = ndarray::Array::from_vec(x_vec);
                x.to_pyarray(py)
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
            )
        }
    }
}

wrap_augmentation_functions!(AmplitudePhasePerturbation);

#[pyclass]
pub struct DynamicTimeWarpAugmenter {
    inner: fraug::augmenters::DynamicTimeWarpAugmenter,
}

#[pymethods]
impl DynamicTimeWarpAugmenter {
    #[new]
    fn new(window_size: usize) -> Self {
        DynamicTimeWarpAugmenter {
            inner: fraug::augmenters::DynamicTimeWarpAugmenter::new(window_size),
        }
    }
}

wrap_augmentation_functions!(DynamicTimeWarpAugmenter);

#[pyclass]
pub struct FrequencyMask {
    inner: fraug::augmenters::FrequencyMask,
}

#[pymethods]
impl FrequencyMask {
    #[new]
    fn new(mask_width: usize) -> Self {
        FrequencyMask {
            inner: fraug::augmenters::FrequencyMask::new(mask_width),
        }
    }
}

wrap_augmentation_functions!(FrequencyMask);

#[pyclass]
pub struct RandomWindowWarpAugmenter {
    inner: fraug::augmenters::RandomWindowWarpAugmenter,
}

#[pymethods]
impl RandomWindowWarpAugmenter {
    #[new]
    fn new(window_size: usize, speed_ratio_range: (f64, f64)) -> Self {
        RandomWindowWarpAugmenter {
            inner: fraug::augmenters::RandomWindowWarpAugmenter::new(
                window_size,
                speed_ratio_range,
            ),
        }
    }
}

wrap_augmentation_functions!(RandomWindowWarpAugmenter);

#[pyclass]
pub enum PoolingMethod {
    Max,
    Min,
    Average
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
            inner: fraug::augmenters::Pool::new(
                int_kind,
                size
            ),
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
            inner: fraug::augmenters::Quantize::new(
                levels
            ),
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
            inner: fraug::augmenters::Resize::new(
                size
            ),
        }
    }
}

wrap_augmentation_functions!(Resize);