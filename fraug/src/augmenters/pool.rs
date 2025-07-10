use super::base::Augmenter;

/// Reduces the temporal resolution without changing the length
pub struct Pool {
    /// Pooling function to be used
    kind: PoolingMethod,
    /// Size of one pool
    size: usize,
}

pub enum PoolingMethod {
    Max,
    Min,
    Average,
}

impl Pool {
    /// Creates new pool augmenter
    pub fn new(kind: PoolingMethod, size: usize) -> Self {
        Pool { kind, size }
    }
}

impl Augmenter for Pool {
    fn augment_one(&self, x: &[f64]) -> Vec<f64> {
        let mut res = Vec::with_capacity(x.len());

        let mut i = 0;
        while i < x.len() {
            let cur_size = if i + self.size < x.len() {
                self.size
            } else {
                x.len() - i
            };

            let new_val = {
                match self.kind {
                    PoolingMethod::Max => *x[i..i + cur_size]
                        .iter()
                        .reduce(|a, b| if a < b { b } else { a })
                        .unwrap(),
                    PoolingMethod::Min => *x[i..i + cur_size]
                        .iter()
                        .reduce(|a, b| if a < b { a } else { b })
                        .unwrap(),
                    PoolingMethod::Average => {
                        x[i..i + cur_size].iter().sum::<f64>() / cur_size as f64
                    }
                }
            };

            for idx in i..i + cur_size {
                res.push(new_val);
            }

            i += self.size;
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_min() {
        let series = vec![1.0; 5]
            .iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect::<Vec<_>>();

        let aug = Pool::new(PoolingMethod::Min, 3);
        let series = aug.augment_one(&series);

        assert_eq!(series, vec![0.0, 0.0, 0.0, 3.0, 3.0]);
    }

    #[test]
    fn pool_max() {
        let series = vec![1.0; 5]
            .iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect::<Vec<_>>();

        let aug = Pool::new(PoolingMethod::Max, 3);
        let series = aug.augment_one(&series);

        assert_eq!(series, vec![2.0, 2.0, 2.0, 4.0, 4.0]);
    }

    #[test]
    fn pool_average() {
        let series = vec![1.0; 6]
            .iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect::<Vec<_>>();

        let aug = Pool::new(PoolingMethod::Average, 4);
        let series = aug.augment_one(&series);

        assert_eq!(series, vec![1.5, 1.5, 1.5, 1.5, 4.5, 4.5]);
    }

    #[test]
    fn pool_exact_match() {
        let series = vec![1.0; 6]
            .iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect::<Vec<_>>();

        let aug = Pool::new(PoolingMethod::Min, 2);
        let series = aug.augment_one(&series);

        assert_eq!(series, vec![0.0, 0.0, 2.0, 2.0, 4.0, 4.0]);
    }
}
