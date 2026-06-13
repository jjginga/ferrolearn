pub mod abalone;

// Computes the arithmetic mean of a slice of values.
// Formula: μ = (1/n) * Σxᵢ
pub fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

// Computes the population standard deviation.
// Formula: σ = sqrt((1/n) * Σ(xᵢ - μ)²)
// We use population std (divide by n) rather than sample std (divide by n-1)
// because we treat the dataset as the full population, not a sample of a larger one.
pub fn std(data: &[f64]) -> f64 {
    let m = mean(data);
    // For each value, compute the squared deviation from the mean,
    // then average them — that's the variance.
    let variance = data.iter()
        .map(|x| (x - m).powi(2))  // (xᵢ - μ)²
        .sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

// Returns the smallest value in the slice.
// Uses fold to scan the entire slice, keeping track of the running minimum.
// Starts at +∞ so the first real value always wins.
pub fn min(data: &[f64]) -> f64 {
    data.iter().cloned().fold(f64::INFINITY, f64::min)
}

// Returns the largest value in the slice.
// Mirror of min — starts at -∞ so the first real value always wins.
pub fn max(data: &[f64]) -> f64 {
    data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

// Computes the Pearson correlation coefficient between two columns.
// Formula: r = Σ((xᵢ - x̄)(yᵢ - ȳ)) / (n * σx * σy)
// Result is in [-1, 1]:
//   1  → perfect positive linear relationship
//   0  → no linear relationship
//  -1  → perfect negative linear relationship
// Assumes x and y have the same length.
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = mean(x);
    let my = mean(y);
    // zip pairs x and y element by element: (x₀,y₀), (x₁,y₁), ...
    // then for each pair compute (xᵢ - x̄)(yᵢ - ȳ) and sum
    let numerator: f64 = x.iter().zip(y.iter())
        .map(|(xi, yi)| (xi - mx) * (yi - my))
        .sum();
    numerator / (n * std(x) * std(y))
}