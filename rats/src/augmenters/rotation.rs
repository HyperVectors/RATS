use super::base::Augmenter;
use tracing::{info_span};
/// Augmenter that rotates the data 180 degrees around `anchor`
pub struct Rotation {
    pub name: String,
    pub anchor: f64,
    p: f64,
}

impl Rotation {
    pub fn new(anchor: f64) -> Self {
        Rotation {
            name: "Rotation".to_string(),
            anchor,
            p: 1.0,
        }
    }
}

impl Augmenter for Rotation {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let span = info_span!("", step = "augment_one");
        let _enter = span.enter();

        x.iter()
            .map(|val| (*val - self.anchor) * -1.0 + self.anchor)
            .collect()
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
