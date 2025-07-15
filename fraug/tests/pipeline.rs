use fraug::Dataset;
use fraug::augmenters::{AugmentationPipeline, Augmenter, Crop, Drop, Repeat};

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
