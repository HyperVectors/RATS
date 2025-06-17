use super::base::Augmenter;

pub fn dtw(a: &[f64], b: &[f64]) -> (f64, Vec<(usize, usize)>) {
    let n = a.len();
    let m = b.len();
    let mut cost = vec![vec![f64::INFINITY; m + 1]; n + 1];
    cost[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let diff = (a[i - 1] - b[j - 1]).abs();
            let prev = cost[i - 1][j]
                .min(cost[i][j - 1])
                .min(cost[i - 1][j - 1]);
            cost[i][j] = diff + prev;
        }
    }
    let distance = cost[n][m];

    let mut path = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        path.push((i.saturating_sub(1), j.saturating_sub(1)));
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let c_diag = cost[i - 1][j - 1];
            let c_up = cost[i - 1][j];
            let c_left = cost[i][j - 1];
            if c_diag <= c_up && c_diag <= c_left {
                i -= 1;
                j -= 1;
            } else if c_up < c_left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
    }
    path.reverse();
    (distance, path)
}