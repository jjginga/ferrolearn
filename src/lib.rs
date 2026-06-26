mod utils;
pub mod data;
pub mod eda;
mod math;
mod ml;
pub mod models;
pub mod wasm;

// Re-export WASM-facing types so wasm-bindgen can find them at the crate root.
pub use wasm::WasmAbalone;
pub use wasm::linear_regression::{WasmLinearRegression, wasm_grid_search, wasm_cv_weights};
pub use wasm::logistic_regression::{WasmLogisticRegression, wasm_grid_search_logistic, wasm_cv_weights_logistic};