use std::env;
mod augmenters;
mod readcsv;
mod transforms; // <-- Add this line
use augmenters::{AddNoise, AugmentationPipeline, Augmenter, ConditionalAugmenter, Crop, Jittering, Repeat, Rotation, Scaling};
use transforms::fastfourier::{dataset_fft, dataset_ifft, compare_datasets_within_tolerance};

pub struct Dataset {
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

    let pipeline = AugmentationPipeline::new() 
                                        + Crop::new(250)
                                        + ConditionalAugmenter::new(Rotation::new(2.0), 0.5)
                                        + Scaling::new(0.5, 2.0);
                                        //+ Jittering::new(0.1);

    pipeline.augment_dataset(&mut data);

    println!(
        "After {:?}\nLength: {}",
        data.features[0].iter().take(10).collect::<Vec<&f64>>(),
        data.features.len()
    );

    // Write augmented dataset to CSV
    let out_filename = format!("{}_augmented.csv", dataset_name);
    if let Err(e) = readcsv::write_dataset_csv(&data.features, &data.labels, dataset_name, &out_filename) {
        eprintln!("Failed to write augmented CSV: {e}");
    } else {
        println!("Augmented dataset written to {out_filename}");
    }





    // FFT transform of the dataset
    let freq_data = dataset_fft(&data);
    println!(
        "First 10 FFT magnitudes of first sample: {:?}",
        freq_data.features[0].iter().take(10).collect::<Vec<&f64>>()
    );

    // frequency domain dataset to CSV
    let freq_out_filename = format!("{}_fft.csv", dataset_name);
    if let Err(e) = readcsv::write_dataset_csv(&freq_data.features, &freq_data.labels, dataset_name, &freq_out_filename) {
        eprintln!("Failed to write frequency domain CSV: {e}");
    } else {
        println!("Frequency domain dataset written to {freq_out_filename}");
    }

    // IFFT (reconstruction) of the frequency domain dataset
    let time_data = dataset_ifft(&freq_data);
    println!(
        "First 10 reconstructed values of first sample: {:?}",
        time_data.features[0].iter().take(10).collect::<Vec<&f64>>()
    );

    // reconstructed time domain dataset to CSV
    let time_out_filename = format!("{}_ifft.csv", dataset_name);
    if let Err(e) = readcsv::write_dataset_csv(&time_data.features, &time_data.labels, dataset_name, &time_out_filename) {
        eprintln!("Failed to write time domain CSV: {e}");
    } else {
        println!("Time domain dataset written to {time_out_filename}");
    }

    // Compare original and reconstructed datasets
    let (max_diff, all_within) = compare_datasets_within_tolerance(&data, &time_data, 1e-10);
    println!(
        "Max absolute difference after FFT->IFFT: {:.3e}, All within tolerance: {}",
        max_diff, all_within
    );


}
