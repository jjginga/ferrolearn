use wasm_bindgen::prelude::*;
use js_sys::Float64Array;

use crate::wasm::WasmAbalone;
use crate::models::logistic_regression::LogisticRegression;
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
    loss_train: Vec<f64>,
    loss_val: Vec<f64>,
    acc_train: Vec<f64>,
    acc_val: Vec<f64>,
}

#[derive(serde::Serialize)]
struct GridSearchEntry {
    lambda: f64,
    score: f64,  // average validation accuracy across folds
}

// Opaque WASM handle wrapping a trained LogisticRegression.
#[wasm_bindgen]
pub struct WasmLogisticRegression {
    model: LogisticRegression,
    feature_names: Vec<String>,
}

#[derive(serde::Serialize)]
struct FoldWeight {
    fold: usize,
    name: String,
    weight: f64,
}

#[wasm_bindgen]
impl WasmLogisticRegression {

    // reg_type: "none" | "l1" | "l2"
    #[wasm_bindgen(constructor)]
    pub fn new(learning_rate: f64, reg_type: &str, lambda: f64, epochs: usize) -> Self {
        let reg = match reg_type {
            "l1" => Regularization::L1(lambda),
            "l2" => Regularization::L2(lambda),
            _    => Regularization::None,
        };
        WasmLogisticRegression {
            model: LogisticRegression::new(learning_rate, reg, epochs),
            feature_names: Vec::new(),
        }
    }

    // Train on the abalone dataset (non-infant samples only).
    // Returns loss and accuracy history for train and val splits — one value per epoch.
    pub fn fit(&mut self, abalone: &WasmAbalone) -> JsValue {
        let x = abalone.dataset.sex_feature_matrix();
        let y = abalone.dataset.sex_targets();
        let m = abalone.dataset.sex_len();
        let n = abalone.dataset.sex_feature_names().len();

        // 80/20 split — first 80% train, last 20% val
        let m_train = (m as f64 * 0.8) as usize;
        let m_val   = m - m_train;

        let x_train = &x[..m_train * n];
        let x_val   = &x[m_train * n..];
        let y_train = &y[..m_train];
        let y_val   = &y[m_train..];

        self.feature_names = abalone.dataset.sex_feature_names()
            .iter().map(|s| s.to_string()).collect();

        self.model.fit_with_val(x_train, y_train, m_train, x_val, y_val, m_val, n);

        serde_wasm_bindgen::to_value(&FitResult {
            loss_train: self.model.loss_train_history.clone(),
            loss_val:   self.model.loss_val_history.clone(),
            acc_train:  self.model.acc_train_history.clone(),
            acc_val:    self.model.acc_val_history.clone(),
        }).unwrap()
    }

    // Returns [{actual, predicted}] for every non-infant sample — predicted is a probability in [0,1].
    pub fn predictions(&self, abalone: &WasmAbalone) -> JsValue {
        let x = abalone.dataset.sex_feature_matrix();
        let y = abalone.dataset.sex_targets();
        let m = abalone.dataset.sex_len();
        let n = abalone.dataset.sex_feature_names().len();

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

    // Accuracy over the full non-infant dataset — fraction of correctly classified samples.
    pub fn accuracy(&self, abalone: &WasmAbalone) -> f64 {
        let x = abalone.dataset.sex_feature_matrix();
        let y = abalone.dataset.sex_targets();
        let m = abalone.dataset.sex_len();
        let n = abalone.dataset.sex_feature_names().len();
        let preds = self.model.predict(&x, m, n);
        preds.iter().zip(y.iter())
            .filter(|(&p, &t)| (p >= 0.5) == (t > 0.5))
            .count() as f64 / m as f64
    }
}

// Grid search for logistic regression — selects best lambda by highest average validation accuracy.
#[wasm_bindgen]
pub fn wasm_grid_search_logistic(
    abalone: &WasmAbalone,
    reg_type: &str,
    k: usize,
    learning_rate: f64,
    epochs: usize,
    lambdas: Float64Array,
) -> JsValue {
    let x = abalone.dataset.sex_feature_matrix();
    let y = abalone.dataset.sex_targets();
    let m = abalone.dataset.sex_len();
    let n = abalone.dataset.sex_feature_names().len();

    let lambdas  = lambdas.to_vec();
    let reg_type = reg_type.to_string();

    let results: Vec<GridSearchEntry> = grid_search(
        &x, &y, m, n, k,
        |lambda| {
            let reg = match reg_type.as_str() {
                "l1" => Regularization::L1(lambda),
                "l2" => Regularization::L2(lambda),
                _    => Regularization::None,
            };
            LogisticRegression::new(learning_rate, reg, epochs)
        },
        &lambdas,
        // accuracy: higher is better — JS picks the max
        |model: &LogisticRegression, x, y, m, n| {
            let preds = model.predict(x, m, n);
            preds.iter().zip(y.iter())
                .filter(|(&p, &t)| (p >= 0.5) == (t > 0.5))
                .count() as f64 / y.len() as f64
        },
    )
        .into_iter()
        .map(|(lambda, score)| GridSearchEntry { lambda, score })
        .collect();

    serde_wasm_bindgen::to_value(&results).unwrap()
}

// Runs k-fold CV with a fixed lambda and returns per-fold weights for the stability boxplot.
#[wasm_bindgen]
pub fn wasm_cv_weights_logistic(
    abalone: &WasmAbalone,
    reg_type: &str,
    k: usize,
    learning_rate: f64,
    epochs: usize,
    lambda: f64,
) -> JsValue {
    let x = abalone.dataset.sex_feature_matrix();
    let y = abalone.dataset.sex_targets();
    let m = abalone.dataset.sex_len();
    let n = abalone.dataset.sex_feature_names().len();
    let feature_names = abalone.dataset.sex_feature_names();

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

        let mut model = LogisticRegression::new(learning_rate, reg, epochs);
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