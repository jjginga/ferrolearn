use std::collections::HashMap;
use crate::data::abalone::AbaloneDataset;
use crate::eda::{mean, std, min, max, correlation};
use serde::Serialize;

// Serialize allows these structs to be converted to JS objects via serde-wasm-bindgen.
// Without it, wasm-bindgen cannot pass them across the Rust/JS boundary.

// Descriptive statistics for a single numerical column.
// One of these is produced for each feature in the dataset.
#[derive(Serialize)]
pub struct ColumnStats {
    pub mean: f64,  // arithmetic mean — μ = (1/n) * Σxᵢ
    pub std: f64,   // population standard deviation — σ = sqrt((1/n) * Σ(xᵢ - μ)²)
    pub min: f64,   // smallest observed value
    pub max: f64,   // largest observed value
}

// Summary statistics for all samples belonging to one sex category (M, F, or I).
// Used to visualise how age (rings) differs between male, female and infant abalone.
#[derive(Serialize)]
pub struct GroupStats {
    pub sex: String,        // "M", "F", or "I"
    pub mean_rings: f64,    // average number of rings in this group
    pub std_rings: f64,     // spread of rings — how consistent is age within the group
    pub count: usize,       // number of samples in this group
}

// Computes mean, std, min, max for each numerical column in the dataset.
// Returns a HashMap mapping column name → ColumnStats.
// Sex is excluded — it is categorical and has no meaningful statistics.
pub fn summary_stats(dataset: &AbaloneDataset) -> HashMap<String, ColumnStats> {
    let mut stats = HashMap::new();

    for (name, data) in dataset.numerical_columns() {
        stats.insert(name.to_string(), ColumnStats {
            mean: mean(&data),
            std:  std(&data),
            min:  min(&data),
            max:  max(&data),
        });
    }

    stats
}

// Splits the dataset by sex and computes mean/std of rings per group.
// Returns one GroupStats per sex category found in the data.
pub fn group_stats_by_sex(dataset: &AbaloneDataset) -> Vec<GroupStats> {
    let mut result = Vec::new();

    let mut categories: Vec<&str> = dataset.sex.iter()
        .map(|s| s.as_str())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    categories.sort();

    for &sex in &categories {
        // Collect rings for all samples matching this sex category
        let rings: Vec<f64> = dataset.sex.iter()
            .zip(dataset.rings.iter())
            .filter(|(s, _)| s.as_str() == sex)
            .map(|(_, &r)| r as f64)
            .collect();

        if rings.is_empty() {
            continue;
        }

        result.push(GroupStats {
            sex: sex.to_string(),
            mean_rings: mean(&rings),
            std_rings:  std(&rings),
            count:      rings.len(),
        });
    }

    result
}

// Computes the full correlation matrix for all numerical columns.
// Returns the column names and a flat Vec<f64> of size n×n in row-major order.
// Row i, column j = correlation between column i and column j.
// The diagonal is always 1.0 (a column is perfectly correlated with itself).
pub fn correlation_matrix(dataset: &AbaloneDataset) -> (Vec<String>, Vec<f64>) {
    let columns = dataset.numerical_columns();
    let n = columns.len();

    // Extract just the names and just the data into separate Vecs
    let names: Vec<String> = columns.iter()
        .map(|(name, _)| name.to_string())
        .collect();

    let data: Vec<&Vec<f64>> = columns.iter()
        .map(|(_, col)| col)
        .collect();

    // Fill the n×n matrix row by row
    // matrix[i * n + j] = correlation between column i and column j
    let mut matrix = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            matrix[i * n + j] = correlation(data[i], data[j]);
        }
    }

    (names, matrix)
}