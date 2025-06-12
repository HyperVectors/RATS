use fraug::Dataset;
use fraug::augmenters::{
    AugmentationPipeline, Augmenter, ConditionalAugmenter, Crop, Drop, Repeat,
};

#[test]
fn combine_two_augmenters() {
    let series = vec![1.0; 100];
    let mut set = Dataset {
        features: vec![series],
        labels: vec![String::from("1")],
    };

    let pipeline = AugmentationPipeline::new() + Repeat::new(5) + Crop::new(20);
    pipeline.augment_dataset(&mut set);

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

    let augmenter = ConditionalAugmenter::new(Drop::new(1.0, None), 0.5);
    augmenter.augment_dataset(&mut set);

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
