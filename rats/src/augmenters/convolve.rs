use super::base::Augmenter;
use rand::{Rng, rng};
use rayon::prelude::*;
use  tracing::{info_span};

/// Enum to specify the kernel window for the `Convolve` augmenter
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
pub struct Convolve {
    pub name: String,
    window: ConvolveWindow,
    size: usize,
    p: f64,
}

impl Convolve {
    pub fn new(window: ConvolveWindow, size: usize) -> Self {
        assert!(size > 0, "Kernel size must be greater than 0");
        Convolve {
            name: "Convolve".to_string(),
            window,
            size,
            p: 1.0,
        }
    }

    fn make_kernel(&self) -> Vec<f64> {
        let n = self.size;
        match self.window {
            ConvolveWindow::Flat => vec![1.0 / n as f64; n],
            ConvolveWindow::Gaussian => {
                let sigma = 0.3 * ((n - 1) as f64) * 0.5 + 0.8;
                let mid = (n as f64 - 1.0) / 2.0;
                let mut kernel = Vec::with_capacity(n);
                for i in 0..n {
                    let x = i as f64 - mid;
                    kernel.push((-0.5 * (x / sigma).powi(2)).exp());
                }
                let sum: f64 = kernel.iter().sum();
                kernel.iter_mut().for_each(|v| *v /= sum);
                kernel
            }
        }
    }

    fn convolve(&self, x: &[f64], kernel: &[f64]) -> Vec<f64> {
        let n = kernel.len();
        let len = x.len();
        if len < n {
            return x.to_vec();
        }
        let mut out = vec![0.0; len];
        let half = n / 2;
        for i in 0..len {
            let mut acc = 0.0;
            for k in 0..n {
                let idx = if i + k >= half && i + k < len + half {
                    i + k - half
                } else {
                    continue;
                };
                if idx < len {
                    acc += x[idx] * kernel[k];
                }
            }
            out[i] = acc;
        }
        out
    }
}

impl Augmenter for Convolve {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let span = info_span!("", step = "augment_one");
        let _enter = span.enter();
        let kernel = self.make_kernel();
        self.convolve(x, &kernel)
    }

    // reimplementing augment_batch to make sure kernel is created only once for each batch
    fn augment_batch(&self, input: &mut crate::Dataset, parallel: bool, _per_sample: bool)
    where
        Self: Sync,
    {
        let kernel = self.make_kernel();
        if parallel {
            input.features.par_iter_mut().for_each(|x| {
                if self.get_probability() > rng().random() {
                    *x = self.convolve(x, &kernel)
                }
            });
        } else {
            input.features.iter_mut().for_each(|x| {
                if self.get_probability() > rng().random() {
                    *x = self.convolve(x, &kernel)
                }
            });
        }
    }

    fn get_probability(&self) -> f64 {
        self.p
    }

    fn set_probability(&mut self, probability: f64) {
        self.p = probability;
    }

    fn get_name(&self) ->String {
        self.name.clone()
    }
}
