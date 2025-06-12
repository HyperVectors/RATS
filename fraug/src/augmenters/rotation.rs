use super::base::Augmenter;

/// Augmenter that rotates the data 180 degrees around specified anchor
pub struct Rotation {
    anchor: f64,
}

impl Rotation {
    pub fn new(anchor: f64) -> Self {
        Rotation { anchor: anchor }
    }
}

impl Augmenter for Rotation {
    fn augment_one(&self, x: &mut [f64]) {
        x.iter_mut()
            .for_each(|val| *val = (*val - self.anchor) * -1.0 + self.anchor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip() {
        let mut series = vec![1.0; 100];

        let augmenter = Rotation::new(0.0);
        augmenter.augment_one(&mut series);

        assert_eq!(series, vec![-1.0; 100]);
    }

    #[test]
    fn anchor() {
        let mut series = vec![1.0; 100];

        let augmenter = Rotation::new(0.5);
        augmenter.augment_one(&mut series);

        assert_eq!(series, vec![0.0; 100]);
    }
}
