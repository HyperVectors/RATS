use super::base::Augmenter;
use crate::Dataset;
use rand_distr::{Normal, Distribution};
use rand::rng;
use std::f64::consts::PI;

/// Amplitude & Phase Perturbation (APP) augmenter.
/// Adds small Gaussian noise to each bin’s magnitude and phase.
pub struct AmplitudePhasePerturbation {
    pub magnitude_std: f64,
    pub phase_std: f64,
}

impl AmplitudePhasePerturbation {
    pub fn new(magnitude_std: f64, phase_std: f64) -> Self {
        Self { magnitude_std, phase_std }
    }

    /// Augment all samples in the dataset in-place
    pub fn augment_dataset(&self, data: &mut Dataset) {
        for sample in data.features.iter_mut() {
            self.augment_one(sample);
        }
    }
}

impl Augmenter for AmplitudePhasePerturbation {
    fn augment_one(&self, x: &mut [f64]) {
        let num_bins = x.len() / 2;
        let mut rng = rng();
        let mag_noise = Normal::new(0.0, self.magnitude_std).unwrap();
        let phase_noise = Normal::new(0.0, self.phase_std).unwrap();

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn test_app_augmenter() {
        let mut data = Dataset {
            features: vec![
                vec![1.0, 0.0].repeat(16),
                vec![2.0, 0.0].repeat(16),
            ],
            labels: vec!["a".to_string(), "b".to_string()],
        };
        let app = AmplitudePhasePerturbation::new(0.1, 0.1);
        let orig = data.features[0].clone();
        app.augment_dataset(&mut data);
        assert_ne!(orig, data.features[0]);
    }
}