use super::base::Augmenter;

/// Augmenter that rotates the data 180 degrees around specified anchor
pub struct Rotation {
    pub name: String,
    anchor: f64,
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
