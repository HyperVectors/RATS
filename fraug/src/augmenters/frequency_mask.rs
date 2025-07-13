use super::base::Augmenter;
use crate::Dataset;
use crate::transforms::fastfourier::{dataset_fft, dataset_ifft};
use rand::{Rng, rng};

pub struct FrequencyMask {
    pub name: String,
    pub mask_width: usize,
    pub is_time_domain: bool,
    p: f64,
}

impl FrequencyMask {
    pub fn new(mask_width: usize, is_time_domain: bool) -> Self {
        FrequencyMask {
            name: "FrequencyMask".to_string(),
            mask_width,
            is_time_domain,
            p: 1.0,
        }
    }
}

impl Augmenter for FrequencyMask {
    fn augment_batch(&self, data: &mut Dataset, _parallel: bool) {
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
        let mut res = x.to_vec();

        let num_bins = x.len() / 2;
        if num_bins < self.mask_width {
            return res;
        }

        let mut rng = rand::rng();
        let center = rng.random_range(self.mask_width / 2..(num_bins - self.mask_width / 2));
        let start = center - self.mask_width / 2;
        let end = start + self.mask_width;
        for bin in start..end {
            let re_idx = 2 * bin;
            let im_idx = 2 * bin + 1;
            res[re_idx] = 0.0;
            res[im_idx] = 0.0;
        }

        res
    }

    fn get_probability(&self) -> f64 {
        self.p
    }

    fn set_probability(&mut self, probability: f64) {
        self.p = probability;
    }
}
