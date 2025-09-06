use super::base::Augmenter;
use tracing::{info_span};
/// Reverses time series
/// 
/// The augmenter turns `[1, 2, 3]` to `[3, 2, 1]`
pub struct Reverse {
    pub name: String,
    p: f64,
}

impl Reverse {
    pub fn new() -> Self {
        Reverse {
            name: "Reverse".to_string(),
            p: 1.0,
        }
    }
}

impl Augmenter for Reverse {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let span = info_span!("", step = "augment_one");
        let _enter = span.enter();

        x.iter().rev().map(|v| *v).collect()
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
