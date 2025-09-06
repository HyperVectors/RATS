use rats::Dataset;
use rats::augmenters::{AugmentationPipeline, Augmenter, Crop, Drop, Repeat, Scaling};

#[test]
fn combine_two_augmenters() {
    let series = vec![1.0; 100];
    let mut set = Dataset {
        features: vec![series],
        labels: vec![String::from("1")],
    };

    let pipeline = AugmentationPipeline::new() + Repeat::new(5) + Crop::new(20);
    pipeline.augment_batch(&mut set, true, false);

    assert_eq!(set.features.len(), 5);
    assert_eq!(set.features[3].len(), 20);
}

#[test]
fn conditional_augmenter() {
    let series = vec![1.0; 100];
    let mut set = Dataset {
        features: vec![series; 100],
        labels: vec![String::from("1")],
    };

    let mut augmenter = Drop::new(1.0, None);
    augmenter.set_probability(0.5);
    augmenter.augment_batch(&mut set, true, false);

    assert_eq!(set.features.len(), 100);
    let mut dropped = 0;
    set.features.iter().for_each(|row| {
        if row.clone() == vec![0.0; 100] {
            dropped += 1;
        } else {
            assert_eq!(row.clone(), vec![1.0; 100]);
        }
    });
    assert!(dropped > 0 && dropped < 100);
}

#[test]
fn per_sample_pipelining() {
    let mut set_per_sample = Dataset {
        features: vec![
            vec![1.0; 10],
            vec![2.0; 10],
            vec![3.0; 10],
        ],
        labels: vec!["a".into(), "b".into(), "c".into()],
    };

    // Pipeline: Scaling (multiply by 2.0) then Crop (length 5)
    let pipeline = AugmentationPipeline::new()
        + Scaling::new(2.0, 2.0)
        + Crop::new(5);

    pipeline.augment_batch(&mut set_per_sample, false, true);

    // Each sample should be scaled and then cropped
    assert_eq!(set_per_sample.features.len(), 3);
    for (i, row) in set_per_sample.features.iter().enumerate() {
        // Original values: 1.0, 2.0, 3.0, after scaling: 2.0, 4.0, 6.0
        let expected = vec![(i as f64 + 1.0) * 2.0; 5];
        assert_eq!(&row[..], &expected[..]);
    }
}