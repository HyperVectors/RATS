use super::base::Augmenter;
use crate::Dataset;
use crate::transforms::fastfourier::{dataset_fft, dataset_ifft};
use rand::{Rng, rng};
use rand_distr::{Distribution, Normal};
use tracing::{info_span};

/// This augmenter perturbs the frequency representation of each time series by adding Gaussian noise
/// to the magnitude and phase of each frequency bin. If `is_time_domain` is true, the input is first
/// transformed to the frequency domain using FFT, the perturbation is applied, and then the result is
/// transformed back to the time domain using IFFT.
/// The standard deviations of the noise for magnitude
/// and phase are controlled by `magnitude_std` and `phase_std`, respectively.
pub struct AmplitudePhasePerturbation {
    pub name: String,
    pub magnitude_std: f64,
    pub phase_std: f64,
    pub is_time_domain: bool,
    p: f64,
}

impl AmplitudePhasePerturbation {
    pub fn new(magnitude_std: f64, phase_std: f64, is_time_domain: bool) -> Self {
        Self {
            name: "AmplitudePhasePerturbation".to_string(),
            magnitude_std,
            phase_std,
            is_time_domain,
            p: 1.0,
        }
    }
}

impl Augmenter for AmplitudePhasePerturbation {
    fn augment_batch(&self, data: &mut Dataset, _parallel: bool, _per_sample: bool) {
        // tracing::info!("Rust: augment_batch called with per_sample = {}", per_sample);
        let span = info_span!("", component = self.get_name());
        let _enter = span.enter();
        if self.is_time_domain {
            let mut transformed_dataset = dataset_fft(data, true);

            transformed_dataset.features.iter_mut().for_each(|sample| {
                if self.get_probability() > rng().random() {
                    *sample = self.augment_one(sample)
                }
            });

            let inverse_dataset = dataset_ifft(&transformed_dataset, true);
            *data = inverse_dataset;
        } else {
            data.features.iter_mut().for_each(|sample| {
                if self.get_probability() > rng().random() {
                    *sample = self.augment_one(sample)
                }
            });
        }
    }

    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let span = info_span!("", step = "augment_one");
        let _enter = span.enter();
        let num_bins = x.len() / 2;
        let mut rng = rng();
        let mag_noise = Normal::new(0.0, self.magnitude_std).unwrap();
        let phase_noise = Normal::new(0.0, self.phase_std).unwrap();

        let mut x = x.to_vec();

        for bin in 0..num_bins {
            let re_idx = 2 * bin;
            let im_idx = 2 * bin + 1;
            let re = x[re_idx];
            let im = x[im_idx];

            // Convert to polar
            let mag = (re * re + im * im).sqrt();
            let phase = im.atan2(re);

            // Add noise
            let mag_perturbed = (mag + mag_noise.sample(&mut rng)).max(0.0);
            let phase_perturbed = phase + phase_noise.sample(&mut rng);

            // Convert back to cartesian
            x[re_idx] = mag_perturbed * phase_perturbed.cos();
            x[im_idx] = mag_perturbed * phase_perturbed.sin();
        }

        x
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

    fn supports_per_sample(&self) -> bool {
        // if in time-domain mode, disable per-sample chaining because of the FFT/IFFT used in the batch
        !self.is_time_domain
    }
    
}
