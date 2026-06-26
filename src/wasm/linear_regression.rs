use wasm_bindgen::prelude::*;
use js_sys::Float64Array;

use crate::wasm::WasmAbalone;
use crate::models::linear_regression::LinearRegression;
use crate::ml::{Regularization, SupervisedModel, grid_search};

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
struct FitResult {
    r2_train: Vec<f64>,
    r2_val: Vec<f64>,
    rmse_train: Vec<f64>,
    rmse_val: Vec<f64>,
}

// One weight entry per feature per fold — used to build the weight stability boxplot in JS.
#[derive(serde::Serialize)]
struct FoldWeight {
    fold: usize,
    name: String,
    weight: f64,
}

#[derive(serde::Serialize)]
struct GridSearchEntry {
    lambda: f64,
    rmse: f64,  // notebook selects best lambda by lowest average validation RMSE
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
    // Returns R² and RMSE history for train and val splits — one value per epoch.
    // JS uses this to plot the evolution curves.
    pub fn fit(&mut self, abalone: &WasmAbalone) -> JsValue {
        let x = abalone.dataset.feature_matrix();
        let y = abalone.dataset.targets();
        let m = abalone.dataset.len();
        let n = abalone.dataset.feature_names().len();

        // 80/20 split — first 80% train, last 20% val
        let m_train = (m as f64 * 0.8) as usize;
        let m_val   = m - m_train;

        // Rows are contiguous in row-major layout: each row is n elements
        let x_train = &x[..m_train * n];
        let x_val   = &x[m_train * n..];
        let y_train = &y[..m_train];
        let y_val   = &y[m_train..];

        self.feature_names = abalone.dataset.feature_names()
            .iter().map(|s| s.to_string()).collect();

        self.model.fit_with_val(x_train, y_train, m_train, x_val, y_val, m_val, n);

        serde_wasm_bindgen::to_value(&FitResult {
            r2_train:   self.model.r2_train_history.clone(),
            r2_val:     self.model.r2_val_history.clone(),
            rmse_train: self.model.rmse_train_history.clone(),
            rmse_val:   self.model.rmse_val_history.clone(),
        }).unwrap()
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

    // R² over the full dataset — how much variance in ring count the model explains.
    pub fn r2(&self, abalone: &WasmAbalone) -> f64 {
        let x = abalone.dataset.feature_matrix();
        let y = abalone.dataset.targets();
        let m = abalone.dataset.len();
        let n = abalone.dataset.feature_names().len();
        self.model.r2(&x, &y, m, n)
    }

    // RMSE over the full dataset — average prediction error in rings (same unit as target).
    pub fn rmse(&self, abalone: &WasmAbalone) -> f64 {
        let x = abalone.dataset.feature_matrix();
        let y = abalone.dataset.targets();
        let m = abalone.dataset.len();
        let n = abalone.dataset.feature_names().len();
        self.model.rmse(&x, &y, m, n)
    }
}

// Grid search as a standalone function — returns [{lambda, rmse}] for all
// candidates so JS can display the full table and highlight the best.
//
// lambdas is passed in from JS as a Float64Array — the search space is defined
// there, not here. This function has no opinion on which values to try.
//
// Trains k * len(lambdas) models — runs synchronously in WASM.
#[wasm_bindgen]
pub fn wasm_grid_search(
    abalone: &WasmAbalone,
    reg_type: &str,
    k: usize,
    learning_rate: f64,
    epochs: usize,
    lambdas: Float64Array,  // dense log-spaced grid generated by JS
) -> JsValue {
    let x = abalone.dataset.feature_matrix();
    let y = abalone.dataset.targets();
    let m = abalone.dataset.len();
    let n = abalone.dataset.feature_names().len();

    let lambdas  = lambdas.to_vec();
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
        |model: &LinearRegression, x, y, m, n| model.rmse(x, y, m, n),
    )
        .into_iter()
        .map(|(lambda, rmse)| GridSearchEntry { lambda, rmse })
        .collect();

    serde_wasm_bindgen::to_value(&results).unwrap()
}

// Runs k-fold CV with a fixed lambda and returns the trained weights from each fold.
// Used to build the weight stability boxplot — shows how much each feature's weight
// varies across folds, which reveals instability caused by multicollinearity.
#[wasm_bindgen]
pub fn wasm_cv_weights(
    abalone: &WasmAbalone,
    reg_type: &str,
    k: usize,
    learning_rate: f64,
    epochs: usize,
    lambda: f64,
) -> JsValue {
    let x = abalone.dataset.feature_matrix();
    let y = abalone.dataset.targets();
    let m = abalone.dataset.len();
    let n = abalone.dataset.feature_names().len();
    let feature_names = abalone.dataset.feature_names();

    let fold_size = m / k;
    let reg_type  = reg_type.to_string();

    let mut entries: Vec<FoldWeight> = Vec::new();

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end   = (fold + 1) * fold_size;

        let mut x_train = Vec::new();
        let mut y_train = Vec::new();

        for i in 0..m {
            let row = &x[i * n..(i + 1) * n];
            if i < test_start || i >= test_end {
                x_train.extend_from_slice(row);
                y_train.push(y[i]);
            }
        }

        let m_train = y_train.len();
        let reg = match reg_type.as_str() {
            "l1" => Regularization::L1(lambda),
            "l2" => Regularization::L2(lambda),
            _    => Regularization::None,
        };

        let mut model = LinearRegression::new(learning_rate, reg, epochs);
        model.fit(&x_train, &y_train, m_train, n);

        // weights[0] is bias — skip; weights[1..] map to feature names
        for (i, name) in feature_names.iter().enumerate() {
            entries.push(FoldWeight {
                fold,
                name:   name.to_string(),
                weight: *model.weights.get(i + 1).unwrap_or(&0.0),
            });
        }
    }

    serde_wasm_bindgen::to_value(&entries).unwrap()
}


