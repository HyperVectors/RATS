use std::env;
mod readcsv;

struct Dataset {
    features: Vec<Vec<f64>>,
    labels: Vec<String>,
}

fn main() {
    // Get dataset name from CLI argument | USAGE : cargo run -- <dataset_name>
    let args: Vec<String> = env::args().collect();
    let dataset_name = if args.len() > 1 {
        &args[1]
    } else {
        "Car"
    };

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
                if !data.features.is_empty() { data.features[0].len() } else { 0 }
            );
        }
        Err(e) => {
            eprintln!("Failed to load dataset '{}': {}", dataset_name, e);
        }
    }

    // Here we can do augmentations to the data

}