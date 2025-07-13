use super::base::Augmenter;
use crate::Dataset;
use rand::rngs::ThreadRng;
use rand::{rng, Rng};

pub struct RandomTimeWarpAugmenter {
    pub name: String,
    /// Length of the window to warp - a window of this size will be selected randomly for every time series in the dataset
    pub window_size: usize,
    /// Range for random speed ratio: [min, max]
    pub speed_ratio_range: (f64, f64),
}

impl RandomTimeWarpAugmenter {
    /// Create a new augmenter with given window size.
    /// `speed_ratio_range` defines the min and max speed change (e.g. (0.5, 2.0)).
    pub fn new(window_size: usize, speed_ratio_range: (f64, f64)) -> Self {
        RandomTimeWarpAugmenter {
            name: "RandomTimeWarpAugmenter".to_string(),
            window_size,
            speed_ratio_range,
        }
    }
   

    fn warp_series(
        series: &[f64],
        speed_ratio_range: (f64, f64),
        rng: &mut ThreadRng,
    )-> Vec<f64>{
        let len = series.len();
        let warp_ratio = rng.random_range(speed_ratio_range.0..=speed_ratio_range.1);

        let mut time_series: Vec<f64> = (0..series.len()).map(|i| i as f64 / warp_ratio).collect();

        if let Some(&max_t) = time_series.last() {
            // need to scale the times to handle the edge case of max of times != len-1. 
            // This breaks the warpng. Hence, iterating and scaling the times
            if max_t > 0.0 {
                let scale = (len - 1) as f64 / max_t;
                for t in &mut time_series {
                    *t *= scale;
                }
            }
        }

        time_series
            .into_iter()
            .map(|t| {
                let floor = t.floor() as usize;
                let ceil = t.ceil() as usize;
                if floor == ceil || ceil >= len {
                    series[floor.min(ceil - 1)]
                } else {
                    let warping_factor = t - floor as f64;
                    series[floor] * (1.0 - warping_factor) + series[ceil] * warping_factor
                }
            })
            .collect()

    }
}

impl Augmenter for RandomTimeWarpAugmenter {
    /// Augment the dataset in-place
    /// For each series, a random contiguous window of length `window_size` is chosen,
    /// then a single pre-defined speed ratio is applied within that window to warp
    fn augment_batch(&self, data:&mut Dataset,_parallel:bool){
        data.features
                .iter_mut()
                .for_each(|sample| *sample = self.augment_one(sample));
    }

    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        //Select a window for every time seri
        let mut rng = rng();
        let mut series = x.to_vec();
        let len= series.len();
        
        
        // Choose a window from the time series to warp. If the length of the window is not given / greater than
        // length of the series, we warp the whole series.
        let (window_start ,window_end) = if self.window_size==0 || self.window_size >= len{
            (0, len)
        }else{
            let start_index = rng.random_range(0..=len - self.window_size);
            (start_index , start_index + self.window_size)
        };
    
        let warped_series = Self::warp_series(&series[window_start..window_end], self.speed_ratio_range, &mut rng);
    
        series[window_start..window_end].copy_from_slice(&warped_series);
    
        series
    }
}
