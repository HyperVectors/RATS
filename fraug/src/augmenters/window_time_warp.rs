use super::base::Augmenter;
use crate::Dataset;
use rand::rngs::ThreadRng;
use rand::{rng, thread_rng, Rng};

pub struct RandomWindowWarpAugmenter {
    /// Length of the window to warp - a window of this size will be selected randomly for every time series in the dataset
    pub window_size: usize,
    /// Range for random speed ratio: [min, max]
    pub speed_ratio_range: (f64, f64),
}

impl RandomWindowWarpAugmenter {
    /// Create a new augmenter with given window size.
    /// `speed_ratio_range` defines the min and max speed change (e.g. (0.5, 2.0)).
    pub fn new(window_size: usize, speed_ratio_range: (f64, f64)) -> Self {
        RandomWindowWarpAugmenter { window_size, speed_ratio_range }
    }

    /// Warp the segment series[start..end] by a random speed ratio within range.
    /// Returns a Vec<f64> of length (end-start).
    fn warp_segment(
        series: &[f64],
        start: usize,
        end: usize,
        speed_range: (f64, f64),
        rng: &mut ThreadRng,
    ) -> Vec<f64> {
        let seg = &series[start..end];
        let seg_len = seg.len();
        let ratio = rng.random_range(speed_range.0..=speed_range.1);
        // Compute warped time positions for each index in segment
        let mut times = Vec::with_capacity(seg_len);
        for i in 0..seg_len {
            // Uniformly stretch/compress: new_time = i / ratio
            times.push((i as f64) / ratio);
        }
        // Normalize times to [0, seg_len - 1]
        if let Some(&last) = times.last() {
            if last > 0.0 {
                let scale = (seg_len - 1) as f64 / last;
                for t in times.iter_mut() { *t *= scale; }
            }
        }
        let mut warped = Vec::with_capacity(seg_len);
        for &t in &times {
            let i0 = t.floor() as usize;
            let i1 = t.ceil() as usize;
            let v = if i0 == i1 || i1 >= seg_len {
                seg[i0.min(seg_len - 1)]
            } else {
                let alpha = t - (i0 as f64);
                seg[i0] * (1.0 - alpha) + seg[i1] * alpha
            };
            warped.push(v);
        }
        warped
    }
}

impl Augmenter for RandomWindowWarpAugmenter {
    /// Augment the dataset in-place by appending one warped series per original.
    /// For each series, a random contiguous window of length `window_size` is chosen,
    /// then a single random speed ratio is applied within that window.
    fn augment_dataset(&self, data: &mut Dataset, _parallel: bool) {
        let originals = data.features.clone();
        let orig_labels = data.labels.clone();
        let mut rng = rng();

        for (series, label) in originals.iter().zip(orig_labels.iter()) {
            let len = series.len();
            if self.window_size >= len {
                // If window is too large, treat whole series
                let warped = Self::warp_segment(series, 0, len, self.speed_ratio_range, &mut rng);
                data.features.push(warped);
                data.labels.push(label.clone());
            } else {
                // Select random window start
                let start = rng.gen_range(0..=len - self.window_size);
                let end = start + self.window_size;
                // Generate warped window segment
                let warped_window = Self::warp_segment(series, start, end, self.speed_ratio_range, &mut rng);
                // Build new series: prefix + warped_window + suffix
                let mut new_series = Vec::with_capacity(len);
                new_series.extend_from_slice(&series[0..start]);
                new_series.extend_from_slice(&warped_window);
                new_series.extend_from_slice(&series[end..len]);
                data.features.push(new_series);
                data.labels.push(label.clone());
            }
        }
    }
    
    fn augment_one(&self, _x: &mut [f64]) {
        unimplemented!("Use augment_dataset instead!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn test_random_window_warp_full_series() {
        let mut data = Dataset {
            features: vec![vec![0.0, 1.0, 2.0, 3.0]],
            labels: vec!["L".into()],
        };
        let aug = RandomWindowWarpAugmenter::new(5, (0.5, 2.0));
        aug.augment_dataset(&mut data, false);
        assert_eq!(data.features.len(), 2);
        assert_eq!(data.features[1].len(), 4);
    }

    #[test]
    fn test_random_window_warp_sub_segment() {
        let orig = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut data = Dataset { features: vec![orig.clone()], labels: vec!["L".into()] };
        let aug = RandomWindowWarpAugmenter::new(2, (0.5, 1.5));
        aug.augment_dataset(&mut data, false);
        assert_eq!(data.features.len(), 2);
        let warped = &data.features[1];
        assert_eq!(warped[0], orig[0]);
        assert_eq!(warped[warped.len()-1], orig[orig.len()-1]);
        assert_eq!(warped.len(), orig.len());
    }
}
