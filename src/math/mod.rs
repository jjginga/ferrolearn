// ─── Matrix operations ────────────────────────────────────────────────────────

// Prepend a column of 1s for the bias term: m×n → m×(n+1)
pub fn augment(x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * (n + 1)];
    for i in 0..m {
        out[i * (n + 1)] = 1.0;
        for j in 0..n {
            out[i * (n + 1) + j + 1] = x[i * n + j];
        }
    }
    out
}

// (m×n) · (n) → (m)
// result[i] = Σⱼ A[i,j] · x[j]
pub fn mat_vec_mul(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..m)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
        .collect()
}

// Aᵀ · x without building the transpose: result[j] = Σᵢ A[i,j] · x[i]
pub fn mat_t_vec_mul(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for i in 0..m {
        for j in 0..n {
            out[j] += a[i * n + j] * x[i];
        }
    }
    out
}

// ─── Normalisation ────────────────────────────────────────────────────────────

// μⱼ = (1/m) Σᵢ x[i,j]
pub fn column_means(x: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| (0..m).map(|i| x[i * n + j]).sum::<f64>() / m as f64)
        .collect()
}

// σⱼ = sqrt((1/m) Σᵢ (x[i,j] − μⱼ)²)
pub fn column_stds(x: &[f64], m: usize, n: usize, means: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let var = (0..m)
                .map(|i| (x[i * n + j] - means[j]).powi(2))
                .sum::<f64>() / m as f64;
            var.sqrt()
        })
        .collect()
}

// x̂[i,j] = (x[i,j] − μⱼ) / σⱼ
// If σⱼ ≈ 0 (constant feature), leave as 0 to avoid division by zero
pub fn normalize(x: &[f64], m: usize, n: usize, means: &[f64], stds: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            if stds[j] > 1e-10 {
                out[i * n + j] = (x[i * n + j] - means[j]) / stds[j];
            }
        }
    }
    out
}