use std::env;
mod augmenters;
mod readcsv;
mod transforms;
use crate::augmenters::NoiseType;
use augmenters::{
    AddNoise, AugmentationPipeline, Augmenter, ConditionalAugmenter, Crop, Drop, Jittering, Repeat,
    Rotation, Scaling, FrequencyMask, AmplitudePhasePerturbation, DynamicTimeWarpAugmenter
};
use fraug::Dataset;
use transforms::fastfourier::{compare_datasets_within_tolerance, dataset_fft, dataset_ifft};

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
    
    println!(
         "Before {:?} Length: {}",
         data.features[0].iter().take(10).collect::<Vec<&f64>>(),
         data.features.len()
     );

    // let pipeline = AugmentationPipeline::new() + AddNoise::new(NoiseType::Slope, Some((0.01, 0.02)), None, None);
    // let pipeline = AugmentationPipeline::new()
    //     + Crop::new(250)
    //     + ConditionalAugmenter::new(Rotation::new(2.0), 0.5)
    //     + Scaling::new(0.5, 2.0)
    //     + AddNoise::new(NoiseType::Spike, Some((-2.0, 2.0)), None, None)
    //     + Drop::new(0.05, None)
    //     + AmplitudePhasePerturbation::new(-10.0, 1.7, true);

    // pipeline.augment_dataset(&mut data);

    
    let dtw_augmenter = DynamicTimeWarpAugmenter:: new(10);

    dtw_augmenter.augment_dataset(&mut data, false);

    println!(
        "After {:?}\nLength: {}",
        data.features[0].iter().take(10).collect::<Vec<&f64>>(),
        data.features.len()
    );
    
    // Write augmented dataset to CSV
    let out_filename = format!("{}_augmented.csv", dataset_name);
    if let Err(e) =
        readcsv::write_dataset_csv(&data.features, &data.labels, dataset_name, &out_filename)
    {
        eprintln!("Failed to write augmented CSV: {e}");
    } else {
        println!("Augmented dataset written to {out_filename}");
    }

    // FFT transform of the dataset
    //let mut freq_data = dataset_fft(&data);
    //println!(
    //    "First 10 FFT magnitudes of first sample: {:?}",
    //    freq_data.features[1].iter().take(10).collect::<Vec<&f64>>()
    //);

    // Apply Amplitude & Phase Perturbation
    // let app = AmplitudePhasePerturbation::new(-10.0, 1.7); // Adjust stddevs as needed
    // app.augment_dataset(&mut data, true);
    
    // reconstructed time domain dataset to CSV
    // let time_out_filename = format!("{}app.csv", dataset_name);
    // if let Err(e) = readcsv::write_dataset_csv(
    //     &data.features,
    //     &data.labels,
    //     dataset_name,
    //     &time_out_filename,
    // ) {
    //     eprintln!("Failed to write time domain CSV: {e}");
    // } else {
    //     println!("Time domain dataset written to {time_out_filename}");
    // }

    // Compare original and reconstructed datasets
    //let (max_diff, all_within) = compare_datasets_within_tolerance(&data, &original_data, 1e-10);
    //println!(
    //    "Max absolute difference after FFT->IFFT: {:.3e}, All within tolerance: {}",
    //    max_diff, all_within
    //);
}
