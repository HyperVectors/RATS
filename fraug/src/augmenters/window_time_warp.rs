use super::base::Augmenter;
use crate::Dataset;
use rand::prelude::*;

pub struct RandomWindowWarpAugmenter {
    /// Length of the window to warp - a windoe of this size will be selected randomnly for every time series in the dataset
    pub window_size: usize,
    /// Range for random speed ratio: [min, max]
    pub speed_ratio_range: (f64, f64),
}