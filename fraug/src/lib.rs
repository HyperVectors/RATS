pub mod augmenters;
pub mod transforms;

pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<String>,
}