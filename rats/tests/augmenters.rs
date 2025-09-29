use rats::Dataset;
use rats::augmenters::{
    AddNoise, AmplitudePhasePerturbation, Augmenter, Crop, FrequencyMask, Jittering, NoiseType,
    Permutate, Pool, PoolingMethod, Quantize, RandomTimeWarpAugmenter, Repeat, Resize, Reverse,
    Rotation, Scaling, Drift, Convolve, ConvolveWindow
};
use rats::quality_benchmarking::dtw;

use rats::transforms::fastfourier::{dataset_fft, dataset_ifft};
use rats::transforms::dct::{dataset_dct, dataset_idct};
use rats::transforms::accuracy::compare_datasets_within_tolerance;

fn make_test_dataset() -> Dataset {
    Dataset {
        features: vec![
            vec![0.0, 1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  7.0],
            vec![1.0, 2.0, 3.0, 4.0,  5.0,  6.0,  7.0,  8.0],
        ],
        labels: vec!["A".into(), "B".into()],
    }
}

#[test]
fn convolve_flat() {
    let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    let augmenter = Convolve::new(ConvolveWindow::Flat, 3);
    let result = augmenter.augment_one(&series);
    
    // Should be different from original (smoothed)
    assert_ne!(result, series);
    // Length should be preserved
    assert_eq!(result.len(), series.len());
}

#[test]
fn convolve_gaussian() {
    let series = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    
    let augmenter = Convolve::new(ConvolveWindow::Gaussian, 3);
    let result = augmenter.augment_one(&series);
    
    // Should be different from original (smoothed)
    assert_ne!(result, series);
    // Length should be preserved
    assert_eq!(result.len(), series.len());
}

#[test]
fn drift_basic() {
    let series = vec![1.0; 100];
    
    let augmenter = Drift::new(0.5, 5);
    let result = augmenter.augment_one(&series);
    
    // Should be different from original
    assert_ne!(result, series);
    // Length should be preserved
    assert_eq!(result.len(), series.len());
}

#[test]
fn drift_zero() {
    let series = vec![1.0, 2.0, 3.0, 4.0];
    
    let augmenter = Drift::new(0.0, 3);
    let result = augmenter.augment_one(&series);
    
    // With zero drift, should be identical
    assert_eq!(result, series);
}

#[test]
fn addnoise_uniform() {
    let series = vec![1.0; 100];

    let augmenter = AddNoise::new(NoiseType::Uniform, Some((-1.0, 1.0)), None, None);
    let series = augmenter.augment_one(&series);

    series
        .iter()
        .for_each(|&val| assert!(val >= 0.0 && val <= 2.0));
    assert_ne!(series, vec![1.0; 100]);
}

#[test]
fn addnoise_gaussian() {
    let series = vec![1.0; 100];

    let augmenter = AddNoise::new(NoiseType::Gaussian, None, Some(0.0), Some(0.5));
    let series = augmenter.augment_one(&series);

    assert_ne!(series, vec![1.0; 100]);
}

#[test]
fn addnoise_spike() {
    let series = vec![1.0; 100];

    let augmenter = AddNoise::new(NoiseType::Spike, Some((-2.0, 2.0)), None, None);
    let series = augmenter.augment_one(&series);

    let mut different = 0;
    series.iter().for_each(|&val| {
        if val != 1.0 {
            different += 1;
        }
    });
    assert_eq!(different, 1);
}

#[test]
fn addnoise_slope() {
    let series = vec![0.0; 100];

    let augmenter = AddNoise::new(NoiseType::Slope, Some((1.0, 2.0)), None, None);
    let series = augmenter.augment_one(&series);

    assert_ne!(series, vec![0.0; 100]);
    assert!(series[99] >= 100.0 && series[99] <= 200.0);
}

#[test]
fn fft_ifft_roundtrip_serial() {
    let orig = make_test_dataset();
    let freq = dataset_fft(&orig, false);
    let recon = dataset_ifft(&freq, false);

    let (max_diff, all_within) =
        compare_datasets_within_tolerance(&orig, &recon, 1e-6);
    assert!(all_within, "FFT to IFFT serial failed, max diff = {}", max_diff);
}

#[test]
fn fft_ifft_roundtrip_parallel() {
    let orig = make_test_dataset();
    let freq = dataset_fft(&orig, true);
    let recon = dataset_ifft(&freq, true);

    let (max_diff, all_within) =
        compare_datasets_within_tolerance(&orig, &recon, 1e-6);
    assert!(all_within, "FFT to IFFT parallel failed, max diff = {}", max_diff);
}

#[test]
fn dct_idct_roundtrip_serial() {
    let orig = make_test_dataset();
    let coeffs = dataset_dct(&orig, false);
    let recon = dataset_idct(&coeffs, false);

    let (max_diff, all_within) =
        compare_datasets_within_tolerance(&orig, &recon, 1e-6);
    assert!(all_within, "DCT to IDCT serial failed, max diff = {}", max_diff);
}

#[test]
fn dct_idct_roundtrip_parallel() {
    let orig = make_test_dataset();
    let coeffs = dataset_dct(&orig, true);
    let recon = dataset_idct(&coeffs, true);

    let (max_diff, all_within) =
        compare_datasets_within_tolerance(&orig, &recon, 1e-6);
    assert!(all_within, "DCT to IDCT parallel failed, max diff = {}", max_diff);
}

#[test]
fn app_augmenter_frequency() {
    let mut data = Dataset {
        features: vec![vec![1.0, 0.0].repeat(16), vec![2.0, 0.0].repeat(16)],
        labels: vec!["a".to_string(), "b".to_string()],
    };
    let app = AmplitudePhasePerturbation::new(0.1, 0.1, false);
    let orig = data.features[0].clone();
    app.augment_batch(&mut data, false, false);
    assert_ne!(orig, data.features[0]);
}

#[test]
fn app_augmenter_time() {
    let mut data = Dataset {
        features: vec![vec![0.0, 1.0, 2.0], vec![0.0, 2.0, 4.0]],
        labels: vec!["A".to_string(), "B".to_string()],
    };
    let orig = data.features[0].clone();

    let app = AmplitudePhasePerturbation::new(0.1, 0.1, true);

    app.augment_batch(&mut data, false, false);

    assert_ne!(orig, data.features[0]);
}

#[test]
fn crop_larger() {
    let series = vec![1.0; 100];
    let mut set = Dataset {
        features: vec![series],
        labels: vec![String::from("1")],
    };

    let augmenter = Crop::new(200);
    augmenter.augment_batch(&mut set, true, false);

    assert_eq!(set.features[0], vec![1.0; 100]);
}

#[test]
fn crop_smaller() {
    let series = vec![1.0; 100];
    let mut set = Dataset {
        features: vec![series],
        labels: vec![String::from("1")],
    };

    let augmenter = Crop::new(50);
    augmenter.augment_batch(&mut set, true, false);

    assert_eq!(set.features[0], vec![1.0; 50]);
}

#[test]
fn drop_all() {
    let series = vec![1.0; 100];

    let drop = rats::augmenters::Drop::new(1.0, None);
    let series = drop.augment_one(&series);

    assert_eq!(series, vec![0.0; 100]);
}

#[test]
fn drop_none() {
    let series = vec![1.0; 100];

    let drop = rats::augmenters::Drop::new(0.0, None);
    let series = drop.augment_one(&series);

    assert_eq!(series, vec![1.0; 100]);
}

#[test]
fn test_frequency_mask_dataset() {
    let mut data = Dataset {
        features: vec![
            vec![1.0, 2.0].repeat(16), // 32 elements: [1.0, 2.0, 1.0, 2.0, ...]
            vec![2.0, 3.0].repeat(16), // 32 elements: [2.0, 3.0, 2.0, 3.0, ...]
        ],
        labels: vec!["a".to_string(), "b".to_string()],
    };
    let mask = FrequencyMask::new(4, false);
    mask.augment_batch(&mut data, true, false);
    for sample in data.features {
        let mut zeroed_bins = 0;
        for bin in 0..(sample.len() / 2) {
            if sample[2 * bin] == 0.0 && sample[2 * bin + 1] == 0.0 {
                zeroed_bins += 1;
            }
        }
        assert!(
            zeroed_bins >= 4,
            "Expected at least 4 zeroed bins, got {}",
            zeroed_bins
        );
    }
}

#[test]
fn jittering() {
    let series = vec![1.0; 100];

    let augmenter = Jittering::new(0.5);
    let series = augmenter.augment_one(&series);

    assert_ne!(series, vec![1.0; 100]);
}

#[test]
fn permutate() {
    let series = vec![1.0, 2.0, 3.0, 4.0];

    let aug = Permutate::new(2, 2);
    let series = aug.augment_one(&series);

    assert!(series == vec![3.0, 4.0, 1.0, 2.0] || series == vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn pool_min() {
    let series = vec![1.0; 5]
        .iter()
        .enumerate()
        .map(|(i, _)| i as f64)
        .collect::<Vec<_>>();

    let aug = Pool::new(PoolingMethod::Min, 3);
    let series = aug.augment_one(&series);

    assert_eq!(series, vec![0.0, 0.0, 0.0, 3.0, 3.0]);
}

#[test]
fn pool_max() {
    let series = vec![1.0; 5]
        .iter()
        .enumerate()
        .map(|(i, _)| i as f64)
        .collect::<Vec<_>>();

    let aug = Pool::new(PoolingMethod::Max, 3);
    let series = aug.augment_one(&series);

    assert_eq!(series, vec![2.0, 2.0, 2.0, 4.0, 4.0]);
}

#[test]
fn pool_average() {
    let series = vec![1.0; 6]
        .iter()
        .enumerate()
        .map(|(i, _)| i as f64)
        .collect::<Vec<_>>();

    let aug = Pool::new(PoolingMethod::Average, 4);
    let series = aug.augment_one(&series);

    assert_eq!(series, vec![1.5, 1.5, 1.5, 1.5, 4.5, 4.5]);
}

#[test]
fn pool_exact_match() {
    let series = vec![1.0; 6]
        .iter()
        .enumerate()
        .map(|(i, _)| i as f64)
        .collect::<Vec<_>>();

    let aug = Pool::new(PoolingMethod::Min, 2);
    let series = aug.augment_one(&series);

    assert_eq!(series, vec![0.0, 0.0, 2.0, 2.0, 4.0, 4.0]);
}

#[test]
fn quantize() {
    let series = vec![1.0; 11]
        .iter()
        .enumerate()
        .map(|(i, _)| i as f64)
        .collect::<Vec<_>>();

    let aug = Quantize::new(5);
    let series = aug.augment_one(&series);

    assert_eq!(
        series,
        vec![0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0]
    );
}

#[test]
fn repeat_2() {
    let series = vec![1.0; 100];
    let mut set = Dataset {
        features: vec![series],
        labels: vec![String::from("1")],
    };

    let augmenter = Repeat::new(2);
    augmenter.augment_batch(&mut set, false, false);

    assert_eq!(set.features[0], vec![1.0; 100]);
    assert_eq!(set.features[1], vec![1.0; 100]);
    assert_eq!(set.features.len(), 2);
    assert_eq!(set.labels, vec![String::from("1"); 2]);
}

#[test]
fn resize_smaller() {
    let series = vec![1.0; 90]
        .iter()
        .enumerate()
        .map(|(i, _)| i as f64)
        .collect::<Vec<_>>();

    let aug = Resize::new(10);
    let series = aug.augment_one(&series);

    assert_eq!(
        series,
        vec![0.0, 9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0, 81.0]
    );
}

#[test]
fn resize_larger() {
    let series = vec![1.0; 5]
        .iter()
        .enumerate()
        .map(|(i, _)| i as f64)
        .collect::<Vec<_>>();

    let aug = Resize::new(12);
    let series = aug.augment_one(&series);

    assert_eq!(
        series,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    );
}

#[test]
fn reverse() {
    let series = vec![1.0, 2.0, 3.0, 4.0];

    let aug = Reverse::new();
    let series = aug.augment_one(&series);

    assert_eq!(series, vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn rotation_flip() {
    let series = vec![1.0; 100];

    let augmenter = Rotation::new(0.0);
    let series = augmenter.augment_one(&series);

    assert_eq!(series, vec![-1.0; 100]);
}

#[test]
fn rotation_anchor() {
    let series = vec![1.0; 100];

    let augmenter = Rotation::new(0.5);
    let series = augmenter.augment_one(&series);

    assert_eq!(series, vec![0.0; 100]);
}

#[test]
fn scaling() {
    let series = vec![1.0; 100];

    let augmenter = Scaling::new(2.0, 4.0);
    let series = augmenter.augment_one(&series);

    series
        .iter()
        .for_each(|&val| assert!(val >= 2.0 && val <= 4.0));
}

#[test]
fn random_time_warp_full_series() {
    let mut data = Dataset {
        features: vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]],
        labels: vec!["L".into(), "L".into()],
    };
    let aug = RandomTimeWarpAugmenter::new(0, (0.5, 2.0));
    aug.augment_batch(&mut data, true, false);
    assert_eq!(data.features.len(), 2);
    for ele in data.features {
        assert_eq!(ele.len(), 4)
    }
}

#[test]
fn random_time_warp_window() {
    let mut data = Dataset {
        features: vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]],
        labels: vec!["L".into(), "L".into()],
    };
    let aug = RandomTimeWarpAugmenter::new(3, (0.5, 2.0));
    aug.augment_batch(&mut data, true, false);
    assert_eq!(data.features.len(), 2);
    for ele in data.features {
        assert_eq!(ele.len(), 4)
    }
}

#[test]
fn random_time_warp_full_series_full_window() {
    let mut data = Dataset {
        features: vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]],
        labels: vec!["L".into(), "L".into()],
    };
    let aug = RandomTimeWarpAugmenter::new(4, (0.5, 2.0));
    aug.augment_batch(&mut data, true, false);
    assert_eq!(data.features.len(), 2);
    for ele in data.features {
        assert_eq!(ele.len(), 4)
    }
}

#[test]
fn time_warp_with_dtw(){
    let mut data = Dataset {
        features: vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]],
        labels: vec!["L".into(), "L".into()],
    };
    let original = data.features.clone();
    let aug = RandomTimeWarpAugmenter::new(2, (0.5, 2.0));
    aug.augment_batch(&mut data, true, true);
    let (distance, _) = dtw(&original[0], &data.features[0]);
    assert_ne!(distance , 0.0);
}   

#[test]
fn time_warp_with_dtw_full_window(){
    let mut data = Dataset {
        features: vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]],
        labels: vec!["L".into(), "L".into()],
    };
    let original = data.features.clone();
    let aug = RandomTimeWarpAugmenter::new(0, (0.5, 2.0));
    aug.augment_batch(&mut data, true, true);
    let (distance, _) = dtw(&original[0], &data.features[0]);
    assert_ne!(distance , 0.0);
}   