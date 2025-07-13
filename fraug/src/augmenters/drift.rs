use super::base::Augmenter;
use rand::Rng;

/// Drifts the value of a time series by a random value at each point in the series.
/// The drift is linear between the points, bounded by `max_drift`.
/// The number of drift points is specified by `n_drift_points`.

pub struct Drift {
    pub name: String,
    pub max_drift: f64,
    pub n_drift_points: usize,
    p: f64,
}

impl Drift {
    /// new drift augmenter
    pub fn new(max_drift: f64, n_drift_points: usize) -> Self {
        Drift {
            name: "Drift".to_string(),
            max_drift,
            n_drift_points: n_drift_points.max(2), // at least 2 points
            p: 1.0,
        }
    }

    fn make_drift(&self, len: usize) -> Vec<f64> {
        let mut rng = rand::rng();
        let n = self.n_drift_points.min(len);
        let mut drift_points = Vec::with_capacity(n);
        for _ in 0..n {
            drift_points.push(rng.random_range(-self.max_drift..=self.max_drift));
        }
        // Linear interpolation between drift points
        let mut drift = vec![0.0; len];
        let seg_len = len as f64 / (n - 1) as f64;
        for i in 0..len {
            let pos = i as f64 / seg_len;
            let left = pos.floor() as usize;
            let right = pos.ceil() as usize;
            let alpha = pos - left as f64;
            let left_val = drift_points[left.min(n - 1)];
            let right_val = drift_points[right.min(n - 1)];
            drift[i] = (1.0 - alpha) * left_val + alpha * right_val;
        }
        drift
    }
}

impl Augmenter for Drift {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let drift = self.make_drift(x.len());
        x.iter().zip(drift.iter()).map(|(xi, di)| xi + di).collect()
    }

    fn get_probability(&self) -> f64 {
        self.p
    }

    fn set_probability(&mut self, probability: f64) {
        self.p = probability;
    }

    fn get_name(&self) ->String {
        self.name.clone()
    }
}
