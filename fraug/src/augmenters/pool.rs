use super::base::Augmenter;

/// Reduces the temporal resolution without changing the length
pub struct Pool {
    pub name: String,
    /// Pooling function to be used
    kind: PoolingMethod,
    /// Size of one pool
    size: usize,
    p: f64,
}

pub enum PoolingMethod {
    Max,
    Min,
    Average,
}

impl Pool {
    /// Creates new pool augmenter
    pub fn new(kind: PoolingMethod, size: usize) -> Self {
        Pool {
            name: "Pool".to_string(),
            kind,
            size,
            p: 1.0,
        }
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

            for _ in i..i + cur_size {
                res.push(new_val);
            }

            i += self.size;
        }

        res
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
