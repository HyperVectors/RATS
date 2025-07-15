use std::env;
mod augmenters;
mod readcsv;
mod transforms;
use crate::augmenters::NoiseType;
use augmenters::{
    AddNoise, AmplitudePhasePerturbation, AugmentationPipeline, Augmenter, Convolve,
    ConvolveWindow, Crop, Drift, Drop, FrequencyMask, Jittering, Repeat,
    Rotation, Scaling,RandomTimeWarpAugmenter,
};
use fraug::{Dataset};
use transforms::fastfourier::{compare_datasets_within_tolerance, dataset_fft, dataset_ifft};
use tracing_subscriber;

fn main() {
    // dataset name from CLI argument | USAGE : cargo run -- <dataset_name>
    tracing_subscriber::fmt::init();
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

    let mut pipeline = AugmentationPipeline::new()
        // + Crop::new(250)
        // + Scaling::new(0.5, 2.0)
        // + AddNoise::new(NoiseType::Spike, Some((-2.0, 2.0)), None, None);
        // + Drop::new(0.05, None);
        // + RandomTimeWarpAugmenter::new(10, (0.5, 0.9));
        // + AmplitudePhasePerturbation::new(-10.0, 1.7, true)
        // + FrequencyMask::new(10, true)
        //+ Convolve::new(ConvolveWindow::Flat, 7)
        + Convolve::new(ConvolveWindow::Gaussian, 31)
        + Drift::new(1.0, 5);
    //     + { let mut a = Drift::new(1.0, 5); a.set_probability(0.5); a };

    pipeline.augment_batch(&mut data, true, true);

    // // let dtw_augmenter = DynamicTimeWarpAugmenter::new(10);

    // // dtw_augmenter.augment_batch(&mut data, false);

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

    // // // FFT transform of the dataset
    // // let mut freq_data = dataset_fft(&data);
    // // println!(
    // //    "First 10 FFT magnitudes of first sample: {:?}",
    // //    freq_data.features[1].iter().take(10).collect::<Vec<&f64>>()
    // // );

    // // // Apply Amplitude & Phase Perturbation
    // // let app = AmplitudePhasePerturbation::new(-10.0, 1.7, false);
    // // app.augment_batch(&mut freq_data, true);

    // // // Apply Frequency Mask
    // // let freq_mask = FrequencyMask::new(10, false);
    // // freq_mask.augment_batch(&mut freq_data, true);
    // // println!(
    // //     "First 10 FFT magnitudes after perturbation and masking: {:?}",
    // //     freq_data.features[1].iter().take(10).collect::<Vec<&f64>>()
    // // );

    // // // write the frequency domain dataset to CSV
    // // let freq_out_filename = format!("{}_freq_augmented.csv", dataset_name);
    // // if let Err(e) = readcsv::write_dataset_csv(
    // //     &freq_data.features,
    // //     &freq_data.labels,
    // //     dataset_name,
    // //     &freq_out_filename,
    // // ) {
    // //     eprintln!("Failed to write frequency domain CSV: {e}");
    // // } else {
    // //     println!("Frequency domain dataset written to {freq_out_filename}");
    // // }

    // // // Inverse FFT to get back to time domain
    // // let mut reconstructed_data = dataset_ifft(&freq_data);
    // // println!(
    // //     "First 10 time domain values after IFFT: {:?}",
    // //     reconstructed_data.features[1].iter().take(10).collect::<Vec<&f64>>()
    // // );

    // // // reconstructed time domain dataset to CSV
    // // let time_out_filename = format!("{}app.csv", dataset_name);
    // // if let Err(e) = readcsv::write_dataset_csv(
    // //     &reconstructed_data.features,
    // //     &reconstructed_data.labels,
    // //     dataset_name,
    // //     &time_out_filename,
    // // ) {
    // //     eprintln!("Failed to write time domain CSV: {e}");
    // // } else {
    // //     println!("Time domain dataset written to {time_out_filename}");
    // // }

    // // // Compare original and reconstructed datasets
    // // let (max_diff, all_within) = compare_datasets_within_tolerance(&data, &reconstructed_data, 1e-6);
    // // println!(
    // //    "Max absolute difference after FFT->IFFT: {:.3e}, All within tolerance: {}",
    // //    max_diff, all_within
    // // );
    
    // let time_warp_augmenter = RandomTimeWarpAugmenter::new(10, (0.5, 0.9));
    // time_warp_augmenter.augment_batch(&mut data, false);

}
