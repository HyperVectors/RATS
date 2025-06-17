use super::base::Augmenter;


pub fn pairwise_dtw(data: &Dataset) -> Vec<Vec<f64>> {
    let n = data.features.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let (dist, _) = dtw(&data.features[i], &data.features[j]);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }
    distances
}