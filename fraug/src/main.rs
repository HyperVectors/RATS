use std::env;
mod augmenters;
mod readcsv;
use augmenters::{AddNoise, Augmenter, Repeat};

struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<String>,
}

fn main() {
    // Get dataset name from CLI argument | USAGE : cargo run -- <dataset_name>
    let args: Vec<String> = env::args().collect();
    let dataset_name = if args.len() > 1 { &args[1] } else { "Car" };

    let mut data = Dataset {
        features: Vec::new(),
        labels: Vec::new(),
    };

    match readcsv::load_dataset(dataset_name) {
        Ok((features, labels)) => {
            data = Dataset { features, labels };
            println!(
                "Loaded dataset '{}': {} samples, {} features per sample",
                dataset_name,
                data.features.len(),
                if !data.features.is_empty() {
                    data.features[0].len()
                } else {
                    0
                }
            );
        }
        Err(e) => {
            eprintln!("Failed to load dataset '{}': {}", dataset_name, e);
        }
    }

    // Here we can do augmentations to the data

    // Just some test augmentations
    println!(
        "Before {:?}",
        data.features[0].iter().take(10).collect::<Vec<&f64>>()
    );

    Repeat::new(5).augment_dataset(&mut data);

    let augmenter = AddNoise::new((-0.1, 0.1));
    augmenter.augment_dataset(&mut data);

    println!(
        "After {:?}\nLength: {}",
        data.features[0].iter().take(10).collect::<Vec<&f64>>(),
        data.features.len()
    );
}
