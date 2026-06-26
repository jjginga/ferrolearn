use wasm_bindgen::prelude::*;

use crate::data::abalone::AbaloneDataset;
use crate::eda::abalone::{summary_stats, group_stats_by_sex, correlation_matrix};

pub mod linear_regression;
pub mod logistic_regression;

#[derive(serde::Serialize)]
struct CorrelationMatrixResult {
    names: Vec<String>,
    matrix: Vec<f64>,
}

// Opaque WASM handle to a parsed AbaloneDataset.
// JS creates one instance with new WasmAbalone(csv) and reuses it.
// Shared across all algorithm demos — each algorithm reads features and targets from it.
#[wasm_bindgen]
pub struct WasmAbalone {
    pub(crate) dataset: AbaloneDataset,
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
