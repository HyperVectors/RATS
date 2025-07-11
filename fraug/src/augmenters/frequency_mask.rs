use super::base::Augmenter;
use rand::Rng;

pub struct FrequencyMask {
    pub name: String,
    pub mask_width: usize,
}

impl FrequencyMask {
    pub fn new(mask_width: usize) -> Self {
        FrequencyMask {
            name: "FrequencyMask".to_string(),
            mask_width,
        }
    }
}

impl Augmenter for FrequencyMask {
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
        // println!("Masking bins {} to {}", start, end);
        for bin in start..end {
            let re_idx = 2 * bin;
            let im_idx = 2 * bin + 1;
            res[re_idx] = 0.0;
            res[im_idx] = 0.0;
        }

        res
    }
}
