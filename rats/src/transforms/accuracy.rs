use crate::Dataset;

/// Computes maximum absolute difference between two Datasets and check if all differences are within a tolerance.
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