mod utils;
pub mod data;
pub mod eda;
mod math;
mod ml;
pub mod models;

use wasm_bindgen::prelude::*;

use crate::data::abalone::AbaloneDataset;
use crate::eda::abalone::{summary_stats, group_stats_by_sex, correlation_matrix};
use crate::models::linear_regression::LinearRegression;
use crate::ml::{Regularization, SupervisedModel, grid_search};

// ─── Linear Regression WASM bindings ─────────────────────────────────────────

#[derive(serde::Serialize)]
struct PredictionPoint {
    actual: f64,
    predicted: f64,
}

#[derive(serde::Serialize)]
struct WeightEntry {
    name: String,
    weight: f64,
}

#[derive(serde::Serialize)]
struct GridSearchEntry {
    lambda: f64,
    mse: f64,
}

#[derive(serde::Serialize)]
struct CorrelationMatrixResult {
    names: Vec<String>,
    matrix: Vec<f64>,
}

// Opaque WASM handle to a parsed AbaloneDataset.
// JS creates one instance with new WasmAbalone(csv) and reuses it.
#[wasm_bindgen]
pub struct WasmAbalone {
    dataset: AbaloneDataset,
}

// Opaque WASM handle wrapping a trained LinearRegression.
// JS constructs it with hyperparameters, calls fit() passing the dataset,
// then reads loss history, predictions, and weights for the demo.
#[wasm_bindgen]
pub struct WasmLinearRegression {
    model: LinearRegression,
    feature_names: Vec<String>,
}

#[wasm_bindgen]
impl WasmLinearRegression {
    // reg_type: "none" | "l1" | "l2"
    #[wasm_bindgen(constructor)]
    pub fn new(learning_rate: f64, reg_type: &str, lambda: f64, epochs: usize) -> Self {
        let reg = match reg_type {
            "l1" => Regularization::L1(lambda),
            "l2" => Regularization::L2(lambda),
            _    => Regularization::None,
        };
        WasmLinearRegression {
            model: LinearRegression::new(learning_rate, reg, epochs),
            feature_names: Vec::new(),
        }
    }

    // Train on the abalone dataset.
    // Returns the loss history as a JS array — one MSE value per epoch.
    // JS animates the loss curve by stepping through this array.
    pub fn fit(&mut self, abalone: &WasmAbalone) -> JsValue {
        let x = abalone.dataset.feature_matrix();
        let y = abalone.dataset.targets();
        let m = abalone.dataset.len();

        self.model.fit(&x, &y, m, abalone.dataset.feature_names().len());

        self.feature_names = abalone.dataset.feature_names()
            .iter().map(|s| s.to_string()).collect();
        serde_wasm_bindgen::to_value(&self.model.loss_history).unwrap()
    }

    // Returns [{actual, predicted}] for every sample — drives the scatter plot.
    pub fn predictions(&self, abalone: &WasmAbalone) -> JsValue {
        let x = abalone.dataset.feature_matrix();
        let y = abalone.dataset.targets();
        let m = abalone.dataset.len();
        let n = abalone.dataset.feature_names().len();

        let preds = self.model.predict(&x, m, n);
        let points: Vec<PredictionPoint> = y.iter().zip(preds.iter())
            .map(|(&actual, &predicted)| PredictionPoint { actual, predicted })
            .collect();

        serde_wasm_bindgen::to_value(&points).unwrap()
    }

    // Returns [{name, weight}] for each feature.
    // weights[0] is bias — skipped here. weights[1..] map to feature names.
    pub fn weights(&self) -> JsValue {
        let entries: Vec<WeightEntry> = self.feature_names.iter().enumerate()
            .map(|(i, name)| WeightEntry {
                name: name.clone(),
                weight: *self.model.weights.get(i + 1).unwrap_or(&0.0),
            })
            .collect();
        serde_wasm_bindgen::to_value(&entries).unwrap()
    }
}

// Grid search as a standalone function — returns [{lambda, mse}] for all
// candidates so JS can display the full table and highlight the best.
//
// Trains k * len(lambdas) models — runs synchronously in WASM.
// For 4177 samples, k=5, 6 lambdas, 500 epochs this takes ~1-2 seconds.
#[wasm_bindgen]
pub fn wasm_grid_search(
    abalone: &WasmAbalone,
    reg_type: &str,
    k: usize,
    learning_rate: f64,
    epochs: usize,
) -> JsValue {
    let x = abalone.dataset.feature_matrix();
    let y = abalone.dataset.targets();
    let m = abalone.dataset.len();
    let n = abalone.dataset.feature_names().len();

    let lambdas = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0];
    let reg_type = reg_type.to_string();  // move into closure

    let results: Vec<GridSearchEntry> = grid_search(
        &x, &y, m, n, k,
        |lambda| {
            let reg = match reg_type.as_str() {
                "l1" => Regularization::L1(lambda),
                "l2" => Regularization::L2(lambda),
                _    => Regularization::None,
            };
            LinearRegression::new(learning_rate, reg, epochs)
        },
        &lambdas,
    )
        .into_iter()
        .map(|(lambda, mse)| GridSearchEntry { lambda, mse })
        .collect();

    serde_wasm_bindgen::to_value(&results).unwrap()
}

#[wasm_bindgen]
impl WasmAbalone {
    // Parses the CSV once and stores the dataset internally.
    #[wasm_bindgen(constructor)]
    pub fn new(csv: &str) -> WasmAbalone {
        WasmAbalone {
            dataset: AbaloneDataset::parse(csv),
        }
    }

    // Returns summary statistics for each numerical column.
    pub fn summary_stats(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&summary_stats(&self.dataset)).unwrap()
    }

    // Returns mean/std of rings grouped by sex (M, F, I).
    pub fn group_stats(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&group_stats_by_sex(&self.dataset)).unwrap()
    }

    // Returns the correlation matrix as { names: [...], matrix: [...] }
    pub fn correlation_matrix(&self) -> JsValue {
        let (names, matrix) = correlation_matrix(&self.dataset);
        serde_wasm_bindgen::to_value(&CorrelationMatrixResult { names, matrix }).unwrap()
    }

    // Returns the number of valid samples parsed from the CSV.
    pub fn sample_count(&self) -> usize {
        self.dataset.len()
    }

    // Returns all values for the target variable (rings) as a JS array.
    pub fn get_rings(&self) -> JsValue {
        let rings: Vec<f64> = self.dataset.rings.iter()
            .map(|&r| r as f64)
            .collect();
        serde_wasm_bindgen::to_value(&rings).unwrap()
    }

    // Returns all values for a named numerical column as a JS array.
    // Column names come from the dataset itself — no hardcoding.
    // Returns an empty array if the name is not recognised.
    pub fn get_column(&self, name: &str) -> JsValue {
        let col = self.dataset.numerical_columns()
            .into_iter()
            .find(|(col_name, _)| *col_name == name)
            .map(|(_, data)| data)
            .unwrap_or_default();
        serde_wasm_bindgen::to_value(&col).unwrap()
    }

    // Returns sex labels ("M", "F", "I") for all samples as a JS array.
    pub fn get_sex(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.dataset.sex).unwrap()
    }
}