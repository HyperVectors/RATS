/// Implementation of Dynamic Time Warping (DTW) algorithm.
/// This function computes the DTW distance between two sequences and returns the distance
/// along with the optimal path.
/// # Arguments
/// * `a` - First sequence as a slice of f64 values.
/// * `b` - Second sequence as a slice of f64 values.
/// # Returns
/// A tuple containing the DTW distance (f64) and a vector of tuples representing the
/// optimal path as pairs of indices (usize, usize).
/// # Examples
/// ```
/// use rats::quality_benchmarking::dtw;
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![2.0, 3.0, 4.0];
/// let (distance, path) = dtw(&a, &b);
/// ```

pub fn dtw(a: &[f64], b: &[f64]) -> (f64, Vec<(usize, usize)>) {
    let n = a.len();
    let m = b.len();
    let mut cost = vec![vec![f64::INFINITY; m + 1]; n + 1];
    cost[0][0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let diff = (a[i - 1] - b[j - 1]).abs();
            let min_prev = cost[i - 1][j].min(cost[i][j - 1]).min(cost[i - 1][j - 1]);
            cost[i][j] = diff + min_prev;
        }
    }
    let distance = cost[n][m];
    // Backtrack to get the best path
    let mut path = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        path.push((i - 1, j - 1));
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let diag = cost[i - 1][j - 1];
            let up = cost[i - 1][j];
            let left = cost[i][j - 1];
            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up < left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
    }
    path.reverse();
    (distance / n as f64, path)
}
