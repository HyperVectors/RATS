use super::base::Augmenter;
use crate::Dataset;
use crate::transforms::fastfourier::{dataset_fft, dataset_ifft};
use rand::rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Amplitude & Phase Perturbation (APP) augmenter.
/// Adds small Gaussian noise to each bin’s magnitude and phase.
pub struct AmplitudePhasePerturbation {
    pub magnitude_std: f64,
    pub phase_std: f64,
    pub is_time_domain: bool,
}

impl AmplitudePhasePerturbation {
    pub fn new(magnitude_std: f64, phase_std: f64, is_time_domain: bool) -> Self {
        Self {
            magnitude_std,
            phase_std,
            is_time_domain,
        }
    }
}

impl Augmenter for AmplitudePhasePerturbation {
    fn augment_dataset(&self, data: &mut Dataset, _parallel: bool) {
        if self.is_time_domain {
            let mut transformed_dataset = dataset_fft(data);
            
            transformed_dataset.features
                .iter_mut()
                .for_each(|sample| *sample = self.augment_one(sample));

            let inverse_dataset = dataset_ifft(&transformed_dataset);
            *data = inverse_dataset;
        } else {
            data.features
                .iter_mut()
                .for_each(|sample| *sample = self.augment_one(sample));
        }
    }

    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn test_app_augmenter_frequency() {
        let mut data = Dataset {
            features: vec![vec![1.0, 0.0].repeat(16), vec![2.0, 0.0].repeat(16)],
            labels: vec!["a".to_string(), "b".to_string()],
        };
        let app = AmplitudePhasePerturbation::new(0.1, 0.1, false);
        let orig = data.features[0].clone();
        app.augment_dataset(&mut data, false);
        assert_ne!(orig, data.features[0]);
    }

    #[test]
    fn test_app_augmenter_time() {
        let mut data = Dataset {
            features: vec![vec![0.0, 1.0, 2.0], vec![0.0, 2.0, 4.0]],
            labels: vec!["A".to_string(), "B".to_string()],
        };
        let orig = data.features[0].clone();

        let app = AmplitudePhasePerturbation::new(0.1, 0.1, true);

        app.augment_dataset(&mut data, false);

        assert_ne!(orig, data.features[0]);
    }
}
