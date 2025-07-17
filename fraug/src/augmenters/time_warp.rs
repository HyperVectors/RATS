use super::base::Augmenter;
use rand::{Rng, rng};
use tracing:: {info, info_span};

/// Augmenter that applied random time warping to the dataset
/// This augmenter randomly selects a window of the time series, specified by the `window_size` argument and applies a speed change to it.
/// The speed change is defined by the `speed_ratio_range` argument, which specifies the minimum and maximum speed ratio.
/// The speed ratio is a multiplier that affects how fast or slow the selected window is stretched or compressed.
/// If the window size is 0 or larger than the time series length, the entire series is warped.
pub struct RandomTimeWarpAugmenter {
    pub name: String,
    /// Length of the window to warp - a window of this size will be selected randomly for every time series in the dataset
    pub window_size: usize,
    /// Range for random speed ratio: [min, max]
    pub speed_ratio_range: (f64, f64),
    p: f64,
}

impl RandomTimeWarpAugmenter {
    /// Create a new augmenter with given window size.
    /// `speed_ratio_range` defines the min and max speed change (e.g. (0.5, 2.0)).
    pub fn new(window_size: usize, speed_ratio_range: (f64, f64)) -> Self {
        RandomTimeWarpAugmenter {
            name: "RandomTimeWarpAugmenter".to_string(),
            window_size,
            speed_ratio_range,
            p: 1.0,
        }
    }

    fn warp_series(series: &[f64],speed_ratio_range: (f64, f64),rng: &mut impl Rng)-> Vec<f64> {
        let len = series.len();
        if len < 2 { return series.to_vec(); }

        // random number between the min anfd max speeed is picked to warp
        let warp_ratio = rng.random_range(speed_ratio_range.0..=speed_ratio_range.1);

        let times: Vec<f64> = (0..len).map(|i| (i as f64) / warp_ratio).collect();

        times.into_iter().map(|t| {
            let t = t.clamp(0.0, (len - 1) as f64);
            let lo = t.floor() as usize;
            let hi = t.ceil()  as usize;
            if lo == hi {
                series[lo]
            } else {
                let w = t - lo as f64;
                series[lo] * (1.0 - w) + series[hi] * w
            }
            }).collect()
        }
}

impl Augmenter for RandomTimeWarpAugmenter {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let span = info_span!("", step = "augment_one");
        let _enter = span.enter();
        let mut rng = rng();
        let mut series = x.to_vec();
        let len = series.len();

        let (window_start, window_end) = if self.window_size == 0 || self.window_size >= len {
            (0, len-1)
        } else {
            let start_index = rng.random_range(0..len - self.window_size);
            (start_index, start_index + self.window_size)
        };
        info!("window selected from : {:?} to {:?} ", window_start, window_end);
        let warped_series = Self::warp_series(
            &series[window_start..=window_end],
            self.speed_ratio_range,
            &mut rng,
        );
        info!("Warped series: {:?}", warped_series);
        series[window_start..=window_end].copy_from_slice(&warped_series);

        series
    }

    fn get_probability(&self) -> f64 {
        self.p
    }

    fn set_probability(&mut self, probability: f64) {
        self.p = probability;
    }

    fn get_name(&self) -> String{
        self.name.clone()
    }
}
