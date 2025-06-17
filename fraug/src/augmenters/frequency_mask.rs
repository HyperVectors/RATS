use super::base::Augmenter;
use crate::Dataset;
use rand::Rng;

pub struct FrequencyMask {
    pub mask_width: usize,
}

impl FrequencyMask {
    pub fn new(mask_width: usize) -> Self {
        FrequencyMask { mask_width }
    }

    /// Augment all samples in the dataset in-place
    pub fn augment_dataset(&self, data: &mut Dataset) {
        for sample in data.features.iter_mut() {
            self.augment_one(sample);
        }
    }
}

impl Augmenter for FrequencyMask {
    fn augment_one(&self, x: &mut [f64]) {
        let num_bins = x.len() / 2;
        if num_bins < self.mask_width {
            return;
        }
        let mut rng = rand::rng();
        let center = rng.random_range(self.mask_width / 2..(num_bins - self.mask_width / 2));
        let start = center - self.mask_width / 2;
        let end = start + self.mask_width;
        println!("Masking bins {} to {}", start, end);
        for bin in start..end {
            let re_idx = 2 * bin;
            let im_idx = 2 * bin + 1;
            x[re_idx] = 0.0;
            x[im_idx] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn test_frequency_mask_dataset() {
        let mut data = Dataset {
            features: vec![
                vec![1.0, 2.0].repeat(16), // 32 elements: [1.0, 2.0, 1.0, 2.0, ...]
                vec![2.0, 3.0].repeat(16), // 32 elements: [2.0, 3.0, 2.0, 3.0, ...]
            ],
            labels: vec!["a".to_string(), "b".to_string()],
        };
        let mask = FrequencyMask::new(4);
        mask.augment_dataset(&mut data);
        for sample in data.features {
            let mut zeroed_bins = 0;
            for bin in 0..(sample.len() / 2) {
                if sample[2 * bin] == 0.0 && sample[2 * bin + 1] == 0.0 {
                    zeroed_bins += 1;
                }
            }
            assert_eq!(zeroed_bins, 4);
        }
    }
}