use crate::Dataset;
use rustfft::{FftPlanner, num_complex::Complex};

use rayon::prelude::*;

pub fn dataset_fft(dataset: &Dataset, parallel: bool) -> Dataset {
    let freq_features: Vec<Vec<f64>> = if parallel {
        dataset.features.par_iter().map(|sample| {
            let len = sample.len();
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(len);
            let mut buffer: Vec<Complex<f64>> =
                sample.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
            fft.process(&mut buffer);
            let mut spectrum = Vec::with_capacity(2 * len);
            for c in buffer {
                spectrum.push(c.re);
                spectrum.push(c.im);
            }
            spectrum
        }).collect()
    } else {
        dataset.features.iter().map(|sample| {
            let len = sample.len();
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(len);
            let mut buffer: Vec<Complex<f64>> =
                sample.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
            fft.process(&mut buffer);
            let mut spectrum = Vec::with_capacity(2 * len);
            for c in buffer {
                spectrum.push(c.re);
                spectrum.push(c.im);
            }
            spectrum
        }).collect()
    };

    Dataset {
        features: freq_features,
        labels: dataset.labels.clone(),
    }
}

/// Reconstruct from [re0, im0, re1, im1, ...] using inverse FFT.
pub fn dataset_ifft(dataset: &Dataset, parallel: bool) -> Dataset {
    let time_features: Vec<Vec<f64>> = if parallel {
        dataset.features.par_iter().map(|sample| {
            let len = sample.len() / 2;
            let mut planner = FftPlanner::new();
            let ifft = planner.plan_fft_inverse(len);
            let mut buffer: Vec<Complex<f64>> = (0..len)
                .map(|i| Complex {
                    re: sample[2 * i],
                    im: sample[2 * i + 1],
                })
                .collect();
            ifft.process(&mut buffer);
            buffer.iter().map(|c| c.re / len as f64).collect()
        }).collect()
    } else {
        dataset.features.iter().map(|sample| {
            let len = sample.len() / 2;
            let mut planner = FftPlanner::new();
            let ifft = planner.plan_fft_inverse(len);
            let mut buffer: Vec<Complex<f64>> = (0..len)
                .map(|i| Complex {
                    re: sample[2 * i],
                    im: sample[2 * i + 1],
                })
                .collect();
            ifft.process(&mut buffer);
            buffer.iter().map(|c| c.re / len as f64).collect()
        }).collect()
    };

    Dataset {
        features: time_features,
        labels: dataset.labels.clone(),
    }
}

/// maximum absolute difference between two Datasets and check if all differences are within a tolerance.
pub fn compare_datasets_within_tolerance(
    original: &Dataset,
    reconstructed: &Dataset,
    tolerance: f64,
) -> (f64, bool) {
    let mut max_diff = 0.0;
    let mut all_within = true;

    for (orig_sample, recon_sample) in original.features.iter().zip(&reconstructed.features) {
        for (&orig, &recon) in orig_sample.iter().zip(recon_sample.iter()) {
            let diff = (orig - recon).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > tolerance {
                all_within = false;
            }
        }
    }

    (max_diff, all_within)
}