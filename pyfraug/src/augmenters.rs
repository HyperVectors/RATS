use numpy::PyArrayMethods;
use numpy::PyArray1;
use ndarray::ArrayViewMut1;
use pyo3::prelude::*;
use crate::Dataset;
use fraug::augmenters::Augmenter;

macro_rules! wrap_augmentation_functions {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            fn augment_dataset(&self, dataset: &mut Dataset, parallel: bool) {
                self.inner.augment_dataset(&mut dataset.inner, parallel);
            }

            fn augment_one<'py>(&self, x: &Bound<'py, PyArray1<f64>>) {
                let mut x: ArrayViewMut1<f64> = unsafe { x.as_array_mut() };
                let mut x_vec = x.as_slice().unwrap().to_vec();

                self.inner.augment_one(&mut x_vec);

                x.as_slice_mut().unwrap().copy_from_slice(x_vec.as_slice());
            }
        }
    };
}

#[pyclass(name="Augmenter", subclass)]
pub struct PyAugmenter {}

#[pymethods]
impl PyAugmenter {
    #[pyo3(signature = (dataset, *, parallel))]
    fn augment_dataset(&self, dataset: &mut Dataset, parallel: bool) {}

    fn augment_one<'py>(&self, _x: &Bound<'py, PyArray1<f64>>) {}
}

#[pyclass(extends=PyAugmenter)]
pub struct Repeat {
    inner: fraug::augmenters::Repeat,
}

#[pymethods]
impl Repeat {
    #[new]
    fn new(times: usize) -> (Self, PyAugmenter) {
        (Repeat { inner: fraug::augmenters::Repeat::new(times) }, PyAugmenter {})
    }
}

 wrap_augmentation_functions!(Repeat);

#[pyclass(extends=PyAugmenter)]
pub struct Scaling {
    inner: fraug::augmenters::Scaling,
}

#[pymethods]
impl Scaling {
    #[new]
    fn new(min: f64, max: f64) -> (Self, PyAugmenter) {
        (Scaling { inner: fraug::augmenters::Scaling::new(min, max) }, PyAugmenter {})
    }
}

wrap_augmentation_functions!(Scaling);

#[pyclass(extends=PyAugmenter)]
pub struct Rotation {
    inner: fraug::augmenters::Rotation,
}

#[pymethods]
impl Rotation {
    #[new]
    fn new(anchor: f64) -> (Self, PyAugmenter) {
        (Rotation { inner: fraug::augmenters::Rotation::new(anchor) }, PyAugmenter {})
    }
}

wrap_augmentation_functions!(Rotation);

#[pyclass(extends=PyAugmenter)]
pub struct Jittering {
    inner: fraug::augmenters::Jittering,
}

#[pymethods]
impl Jittering {
    #[new]
    fn new(standard_deviation: f64) -> (Self, PyAugmenter) {
        (Jittering { inner: fraug::augmenters::Jittering::new(standard_deviation) }, PyAugmenter {})
    }
}

wrap_augmentation_functions!(Jittering);

#[pyclass(extends=PyAugmenter)]
pub struct Drop {
    inner: fraug::augmenters::Drop,
}

#[pymethods]
impl Drop {
    #[new]
    #[pyo3(signature = (percentage, *, default=None))]
    fn new(percentage: f64, default: Option<f64>) -> (Self, PyAugmenter) {
        (Drop { inner: fraug::augmenters::Drop::new(percentage, default) }, PyAugmenter {})
    }
}

wrap_augmentation_functions!(Drop);

#[pyclass(extends=PyAugmenter)]
pub struct Crop {
    inner: fraug::augmenters::Crop,
}

#[pymethods]
impl Crop {
    #[new]
    fn new(size: usize) -> (Self, PyAugmenter) {
        (Crop { inner: fraug::augmenters::Crop::new(size) }, PyAugmenter {})
    }
}

wrap_augmentation_functions!(Crop);

#[pyclass]
pub enum NoiseType {
    Uniform,
    Gaussian,
    Spike,
    Slope
}

#[pyclass(extends=PyAugmenter)]
pub struct AddNoise {
    inner: fraug::augmenters::AddNoise,
}

#[pymethods]
impl AddNoise {
    #[new]
    #[pyo3(signature = (noise_type, *, bounds=None, mean=None, std_dev=None))]
    fn new(noise_type: &NoiseType, bounds: Option<(f64, f64)>, mean: Option<f64>, std_dev: Option<f64>) -> (Self, PyAugmenter) {
        let int_noise_type = match noise_type {
            NoiseType::Uniform => fraug::augmenters::NoiseType::Uniform,
            NoiseType::Gaussian => fraug::augmenters::NoiseType::Gaussian,
            NoiseType::Spike => fraug::augmenters::NoiseType::Spike,
            NoiseType::Slope => fraug::augmenters::NoiseType::Slope
        };

        (AddNoise { inner: fraug::augmenters::AddNoise::new(int_noise_type, bounds, mean, std_dev) }, PyAugmenter {})
    }
}

wrap_augmentation_functions!(AddNoise);

// #[pyclass(extends=PyAugmenter)]
// pub struct ConditionalAugmenter {
//     augmenter: PyAugmenter,
//     probability: f64
// }
//
// #[pymethods]
// impl ConditionalAugmenter {
//     #[new]
//     fn new<'py>(augmenter: Bound<'py, PyAugmenter>, probability: f64) -> (Self, PyAugmenter) {
//         let augmenter: PyAugmenter = augmenter.extract().unwrap();
//         (ConditionalAugmenter { augmenter, probability }, PyAugmenter {})
//     }
//
//     // fn augment_dataset(&self, input: &mut Dataset) {
//     //     let mut rng = rand::rng();
//     //     input.inner.features.iter_mut().for_each(|x| {
//     //         if rng.random::<f64>() < self.probability {
//     //             self.augmenter.inner.augment_one(x)
//     //         }
//     //     });
//     // }
//
//     fn augment_one<'py>(&self, x: &Bound<'py, PyArray1<f64>>) {
//         if rand::random::<f64>() < self.probability {
//             self.augmenter.augment_one(x);
//         }
//     }
// }

// wrap_augmentation_functions!(ConditionalAugmenter);