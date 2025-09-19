use crate::Dataset;
use rats::augmenters::Augmenter;
use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

macro_rules! wrap_augmentation_functions {
    ($struct_name:ident) => {
        #[gen_stub_pymethods]
        #[pymethods]
        impl $struct_name {
            /// Augment a whole batch
            ///
            /// Parallelized when `parallell` is set
            fn augment_batch(&self, dataset: &mut Dataset, parallel: bool) {
                self.inner.augment_batch(&mut dataset.inner, parallel, false);
            }

            /// Augment one time series
            ///
            /// When called, the augmenter will always augment the series no matter what the probability for this augmenter is
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

            /// By setting a probability with this function the augmenter will only augment
            /// a series in a batch with the specified probability
            #[setter]
            fn set_probability(&mut self, p: f64) -> PyResult<()> {
                self.inner.set_probability(p);
                Ok(())
            }
        }
    };
}

/// Augmenter that repeats all data rows `n` times
///
/// Resource intensive because the data needs to be copied `n` times
///
/// Only works with `augment_batch` because the data needs to be cloned
#[gen_stub_pyclass]
#[pyclass]
pub struct Repeat {
    inner: rats::augmenters::Repeat,
}

#[pymethods]
impl Repeat {
    #[new]
    fn new(times: usize) -> Self {
        Repeat {
            inner: rats::augmenters::Repeat::new(times),
        }
    }
}

wrap_augmentation_functions!(Repeat);

/// Augmenter that scales a time series with a random scalar within the range specified
/// by `min_factor` (inclusive) and `max_factor` (inclusive)
#[gen_stub_pyclass]
#[pyclass]
pub struct Scaling {
    inner: rats::augmenters::Scaling,
}

#[pymethods]
impl Scaling {
    #[new]
    fn new(min: f64, max: f64) -> Self {
        Scaling {
            inner: rats::augmenters::Scaling::new(min, max),
        }
    }
}

wrap_augmentation_functions!(Scaling);

/// Augmenter that rotates the data 180 degrees around `anchor`
#[gen_stub_pyclass]
#[pyclass]
pub struct Rotation {
    inner: rats::augmenters::Rotation,
}

#[pymethods]
impl Rotation {
    #[new]
    fn new(anchor: f64) -> Self {
        Rotation {
            inner: rats::augmenters::Rotation::new(anchor),
        }
    }
}

wrap_augmentation_functions!(Rotation);

/// Augmenter that adds white gaussian noise of the specified standard deviation and a mean of 0
///
/// A special case of the `AddNoise` augmenter
#[gen_stub_pyclass]
#[pyclass]
pub struct Jittering {
    inner: rats::augmenters::Jittering,
}

#[pymethods]
impl Jittering {
    #[new]
    fn new(standard_deviation: f64) -> Self {
        Jittering {
            inner: rats::augmenters::Jittering::new(standard_deviation),
        }
    }
}

wrap_augmentation_functions!(Jittering);

/// Augmenter that drops data points in series
///
/// Drops `percentage` % of data points and replaces them with `default`
///
/// When omitted `default = 0`
#[gen_stub_pyclass]
#[pyclass]
pub struct Drop {
    inner: rats::augmenters::Drop,
}

#[pymethods]
impl Drop {
    #[new]
    #[pyo3(signature = (percentage, *, default=None))]
    fn new(percentage: f64, default: Option<f64>) -> Self {
        Drop {
            inner: rats::augmenters::Drop::new(percentage, default),
        }
    }
}

wrap_augmentation_functions!(Drop);

/// Augmenter that crops each series into a random continuous slice of specified `size`
///
/// Also known as window slicing
#[gen_stub_pyclass]
#[pyclass]
pub struct Crop {
    inner: rats::augmenters::Crop,
}

#[pymethods]
impl Crop {
    #[new]
    fn new(size: usize) -> Self {
        Crop {
            inner: rats::augmenters::Crop::new(size),
        }
    }
}

wrap_augmentation_functions!(Crop);

/// Enum to specify the noise type for the AddNoise augmenter
#[gen_stub_pyclass_enum]
#[pyclass]
pub enum NoiseType {
    Uniform,
    Gaussian,
    Spike,
    Slope,
}

/// Augmenter that allows different types of noise injection
///
/// Noise types:
///
/// - Uniform: Adds uniform noise within the given bounds given through the parameter `bounds`
///
/// - Gaussian: Adds gaussian noise with the specified mean and standard deviation according to the corresponding parameters
///
/// - Spike: Adds a spike in the series with a random magnitude (in the range specified by `bounds` of the standard deviation of the original time series
///
/// - Slope: Adds a linear slope trend to the series with a random slope in the range specified by `bounds`
#[gen_stub_pyclass]
#[pyclass]
pub struct AddNoise {
    inner: rats::augmenters::AddNoise,
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
            NoiseType::Uniform => rats::augmenters::NoiseType::Uniform,
            NoiseType::Gaussian => rats::augmenters::NoiseType::Gaussian,
            NoiseType::Spike => rats::augmenters::NoiseType::Spike,
            NoiseType::Slope => rats::augmenters::NoiseType::Slope,
        };

        AddNoise {
            inner: rats::augmenters::AddNoise::new(int_noise_type, bounds, mean, std_dev),
        }
    }
}

wrap_augmentation_functions!(AddNoise);

/// This augmenter perturbs the frequency representation of each time series by adding Gaussian noise
/// to the magnitude and phase of each frequency bin. 
/// 
/// If `is_time_domain` is true, the input is first
/// transformed to the frequency domain using FFT, the perturbation is applied, and then the result is
/// transformed back to the time domain using IFFT.
/// 
/// The standard deviations of the noise for magnitude
/// and phase are controlled by `magnitude_std` and `phase_std`, respectively.
#[gen_stub_pyclass]
#[pyclass]
pub struct AmplitudePhasePerturbation {
    inner: rats::augmenters::AmplitudePhasePerturbation,
}

#[pymethods]
impl AmplitudePhasePerturbation {
    #[new]
    fn new(magnitude_std: f64, phase_std: f64, is_time_domain: bool) -> Self {
        AmplitudePhasePerturbation {
            inner: rats::augmenters::AmplitudePhasePerturbation::new(
                magnitude_std,
                phase_std,
                is_time_domain,
            ),
        }
    }
}

wrap_augmentation_functions!(AmplitudePhasePerturbation);

/// This augmenter applies a frequency-domain mask to each time series, zeroing out a contiguous block of frequency bins.
/// 
/// - If `is_time_domain` is true, the input is first transformed to the frequency domain using FFT, the mask is applied, and then the result is transformed back to the time domain using IFFT.
/// 
/// The width of the mask is controlled by `mask_width`, and the masked region is chosen randomly for each sample.
#[gen_stub_pyclass]
#[pyclass]
pub struct FrequencyMask {
    inner: rats::augmenters::FrequencyMask,
}

#[pymethods]
impl FrequencyMask {
    #[new]
    fn new(mask_width: usize, is_time_domain: bool) -> Self {
        FrequencyMask {
            inner: rats::augmenters::FrequencyMask::new(mask_width, is_time_domain),
        }
    }
}

wrap_augmentation_functions!(FrequencyMask);

/// Augmenter that applies random time warping to the dataset
/// This augmenter randomly selects a window of the time series, specified by the `window_size` argument and applies a speed change to it.
/// The speed change is defined by the `speed_ratio_range` argument, which specifies the minimum and maximum speed ratio.
/// The speed ratio is a multiplier that affects how fast or slow the selected window is stretched or compressed.
/// If the window size is 0 or larger than the time series length, the entire series is warped.
#[gen_stub_pyclass]
#[pyclass]
pub struct RandomTimeWarpAugmenter {
    inner: rats::augmenters::RandomTimeWarpAugmenter,
}

#[pymethods]
impl RandomTimeWarpAugmenter {
    #[new]
    fn new(window_size: usize, speed_ratio_range: (f64, f64)) -> Self {
        RandomTimeWarpAugmenter {
            inner: rats::augmenters::RandomTimeWarpAugmenter::new(window_size, speed_ratio_range),
        }
    }
}

wrap_augmentation_functions!(RandomTimeWarpAugmenter);

/// Enum to specify the pooling function for the `Pool` augmenter
#[gen_stub_pyclass_enum]
#[pyclass]
pub enum PoolingMethod {
    Max,
    Min,
    Average,
}

/// Reduces the temporal resolution without changing the length by pooling multiple samples together
#[gen_stub_pyclass]
#[pyclass]
pub struct Pool {
    inner: rats::augmenters::Pool,
}

#[pymethods]
impl Pool {
    #[new]
    fn new(kind: &PoolingMethod, size: usize) -> Self {
        let int_kind = match kind {
            PoolingMethod::Max => rats::augmenters::PoolingMethod::Max,
            PoolingMethod::Min => rats::augmenters::PoolingMethod::Min,
            PoolingMethod::Average => rats::augmenters::PoolingMethod::Average,
        };

        Pool {
            inner: rats::augmenters::Pool::new(int_kind, size),
        }
    }
}

wrap_augmentation_functions!(Pool);

/// Quantize time series to a level set
///
/// The level set is constructed by uniformly discretizing the range of all values in the series
#[gen_stub_pyclass]
#[pyclass]
pub struct Quantize {
    inner: rats::augmenters::Quantize,
}

#[pymethods]
impl Quantize {
    #[new]
    fn new(levels: usize) -> Self {
        Quantize {
            inner: rats::augmenters::Quantize::new(levels),
        }
    }
}

wrap_augmentation_functions!(Quantize);

/// Changes temporal resolution of time series by changing the length
///
/// Does not interpolate values!
#[gen_stub_pyclass]
#[pyclass]
pub struct Resize {
    inner: rats::augmenters::Resize,
}

#[pymethods]
impl Resize {
    #[new]
    fn new(size: usize) -> Self {
        Resize {
            inner: rats::augmenters::Resize::new(size),
        }
    }
}

wrap_augmentation_functions!(Resize);

/// Reverses time series
///
/// The augmenter turns `[1, 2, 3]` to `[3, 2, 1]`
#[gen_stub_pyclass]
#[pyclass]
pub struct Reverse {
    inner: rats::augmenters::Reverse,
}

#[pymethods]
impl Reverse {
    #[new]
    fn new() -> Self {
        Reverse {
            inner: rats::augmenters::Reverse::new(),
        }
    }
}

wrap_augmentation_functions!(Reverse);

/// Permutate time series
///
/// First, slices each series into segments and then rearranges them randomly
#[gen_stub_pyclass]
#[pyclass]
pub struct Permutate {
    inner: rats::augmenters::Permutate,
}

#[pymethods]
impl Permutate {
    #[new]
    fn new(window_size: usize, segment_size: usize) -> Self {
        Permutate {
            inner: rats::augmenters::Permutate::new(window_size, segment_size),
        }
    }
}

wrap_augmentation_functions!(Permutate);

/// Drifts the value of a time series by a random value at each point in the series.
///
/// The drift is linear between the points, bounded by `max_drift`.
///
/// The number of drift points is specified by `n_drift_points`.
#[gen_stub_pyclass]
#[pyclass]
pub struct Drift {
    inner: rats::augmenters::Drift,
}

#[pymethods]
impl Drift {
    #[new]
    fn new(max_drift: f64, n_drift_points: usize) -> Self {
        Drift {
            inner: rats::augmenters::Drift::new(max_drift, n_drift_points),
        }
    }
}

wrap_augmentation_functions!(Drift);

/// Enum to specify the kernel window for the `Convolve` augmenter
#[gen_stub_pyclass_enum]
#[pyclass]
pub enum ConvolveWindow {
    Flat,
    Gaussian,
}

/// Usage of this augmenter is to convolve time series data with a kernel
///
/// The kernel can be flat or Gaussian, and the size of the kernel are the parameters
///
/// The convolve operation is applied to each time series in the dataset, and smoothening is achieved
/// by averaging the values in the kernel window over the time series data.
#[gen_stub_pyclass]
#[pyclass]
pub struct Convolve {
    inner: rats::augmenters::Convolve,
}

#[pymethods]
impl Convolve {
    #[new]
    fn new(window: &ConvolveWindow, size: usize) -> Self {
        let int_window = match window {
            ConvolveWindow::Flat => rats::augmenters::ConvolveWindow::Flat,
            ConvolveWindow::Gaussian => rats::augmenters::ConvolveWindow::Gaussian,
        };
        Convolve {
            inner: rats::augmenters::Convolve::new(int_window, size),
        }
    }
}

wrap_augmentation_functions!(Convolve);
