use super::base::Augmenter;
use rand::{distr::Uniform, prelude::*, random_range};
use rand_distr::Normal;

/// Augmenter that allows different types of noise injection
///
/// Different noise types:
///     - Uniform: Adds uniform noise within the given bounds given through the parameter `bounds`
///     - Gaussian: Adds gaussian noise with the specified mean and standard deviation according to the corresponding parameters
///     - Spike: Adds a spike in the series with a random magnitude (in the range specified by `bounds` of the standard deviation of the original time series
///     - Slope: Adds a linear slope trend to the series with a random slope in the range specified by `bounds`
pub struct AddNoise {
    noise_type: NoiseType,
    bounds: Option<(f64, f64)>,
    mean: Option<f64>,
    std_dev: Option<f64>,
}

pub enum NoiseType {
    Uniform,
    Gaussian,
    Spike,
    Slope,
}

impl AddNoise {
    pub fn new(
        noise_type: NoiseType,
        bounds: Option<(f64, f64)>,
        mean: Option<f64>,
        std_dev: Option<f64>,
    ) -> Self {
        AddNoise {
            noise_type: noise_type,
            bounds: bounds,
            mean: mean,
            std_dev: std_dev,
        }
    }
}

impl Augmenter for AddNoise {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        match self.noise_type {
            NoiseType::Uniform => {
                let bounds = self.bounds.expect("Bounds not specified");

                let mut rng = rand::rng();
                let dist = Uniform::new(bounds.0, bounds.1)
                    .expect("Couldn't create uniform distribution from specified bounds");
                x.iter().map(|val| *val + dist.sample(&mut rng)).collect()
            }
            NoiseType::Gaussian => {
                let mean = self.mean.expect("Mean not specified");
                let std_dev = self.std_dev.expect("Standard deviation not specified");

                let mut rng = rand::rng();
                let dist = Normal::new(mean, std_dev)
                    .expect("Couldn't create normal distribution from specified mean and standard deviation");
                x.iter().map(|val| *val + dist.sample(&mut rng)).collect()
            }
            NoiseType::Spike => {
                let bounds = self.bounds.expect("Bounds not specified");

                // Calculate std dev of x
                let n = x.len() as f64;
                let mean = x.iter().sum::<f64>() / n;
                let std_dev = (x.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / n).sqrt();

                // Add spike in random location with random magnitude
                let idx: usize = random_range(0..n as usize);
                let magnitude: f64 = random_range(bounds.0..bounds.1);

                let mut res = x.to_vec();
                res[idx] = magnitude * std_dev;
                res
            }
            NoiseType::Slope => {
                let bounds = self.bounds.expect("Bounds not specified");

                let slope: f64 = random_range(bounds.0..bounds.1);
                x.iter()
                    .enumerate()
                    .map(|(i, val)| *val + i as f64 * slope)
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform() {
        let series = vec![1.0; 100];

        let augmenter = AddNoise::new(NoiseType::Uniform, Some((-1.0, 1.0)), None, None);
        let series = augmenter.augment_one(&series);

        series
            .iter()
            .for_each(|&val| assert!(val >= 0.0 && val <= 2.0));
        assert_ne!(series, vec![1.0; 100]);
    }

    #[test]
    fn gaussian() {
        let series = vec![1.0; 100];

        let augmenter = AddNoise::new(NoiseType::Gaussian, None, Some(0.0), Some(0.5));
        let series = augmenter.augment_one(&series);

        assert_ne!(series, vec![1.0; 100]);
    }

    #[test]
    fn spike() {
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
    fn slope() {
        let series = vec![0.0; 100];

        let augmenter = AddNoise::new(NoiseType::Slope, Some((1.0, 2.0)), None, None);
        let series = augmenter.augment_one(&series);

        assert_ne!(series, vec![0.0; 100]);
        assert!(series[99] >= 100.0 && series[99] <= 200.0);
    }
}
