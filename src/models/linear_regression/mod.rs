use crate::math::{augment, mat_vec_mul, mat_t_vec_mul, column_means, column_stds, normalize};
use crate::ml::Regularization;
use crate::ml::SupervisedModel;
pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub feature_mean: Vec<f64>,
    pub feature_std: Vec<f64>,
    learning_rate: f64,
    regularization: Regularization,
    pub epochs: usize,
    pub r2_train_history: Vec<f64>,
    pub r2_val_history: Vec<f64>,
}

impl LinearRegression {
    pub fn new(learning_rate: f64, regularization: Regularization, epochs: usize) -> Self {
        Self {
            weights: Vec::new(),
            feature_mean: Vec::new(),
            feature_std: Vec::new(),
            learning_rate,
            regularization,
            epochs,
            r2_train_history: Vec::new(),
            r2_val_history: Vec::new(),
        }
    }

    pub fn fit_with_val(
        &mut self,
        x_train: &[f64], y_train: &[f64], m_train: usize,
        x_val: &[f64],   y_val: &[f64],   m_val: usize,
        n: usize,
    ) {
        // Compute normalization stats from training data only — never touch val stats
        self.feature_mean = column_means(x_train, m_train, n);
        self.feature_std  = column_stds(x_train, m_train, n, &self.feature_mean);

        // Normalize both sets using training stats
        let x_train_norm = normalize(x_train, m_train, n, &self.feature_mean, &self.feature_std);
        let x_val_norm   = normalize(x_val,   m_val,   n, &self.feature_mean, &self.feature_std);

        // Augment both with bias column
        let x_train_aug = augment(&x_train_norm, m_train, n);
        let x_val_aug   = augment(&x_val_norm,   m_val,   n);
        let n_aug = n + 1;

        self.weights = vec![0.0; n_aug];
        self.r2_train_history    = Vec::with_capacity(self.epochs);
        self.r2_val_history      = Vec::with_capacity(self.epochs);

        // gradient of (1/m)||Xw-y||² is (1/m)·Xᵀe — matching the notebook (no factor of 2)
        let scale = 1.0 / m_train as f64;

        for _ in 0..self.epochs {
            let preds = mat_vec_mul(&x_train_aug, &self.weights, m_train, n_aug);

            let residuals: Vec<f64> = preds.iter()
                .zip(y_train.iter())
                .map(|(p, t)| p - t)
                .collect();

            // Track R² on training set before weight update (matches notebook — R² is the primary metric)
            self.r2_train_history.push(crate::ml::r2(y_train, &preds));

            // Record val R² using current weights
            let val_preds = mat_vec_mul(&x_val_aug, &self.weights, m_val, n_aug);
            self.r2_val_history.push(crate::ml::r2(y_val, &val_preds));

            // Gradient and weight update — identical to fit()
            let mut grad = mat_t_vec_mul(&x_train_aug, &residuals, m_train, n_aug);
            for g in grad.iter_mut() { *g *= scale; }

            match &self.regularization {
                Regularization::L2(lambda) => {
                    for j in 1..n_aug { grad[j] += 2.0 * lambda * self.weights[j]; }
                }
                Regularization::L1(lambda) => {
                    for j in 1..n_aug { grad[j] += lambda * self.weights[j].signum(); }
                }
                Regularization::None => {}
            }

            for (w, g) in self.weights.iter_mut().zip(grad.iter()) {
                *w -= self.learning_rate * g;
            }
        }
    }

}

impl SupervisedModel for LinearRegression {

    // w̃ ← w̃ − α · [(1/m) X̃ᵀ(X̃w̃ − y) + regularization_term]
    // gradient of (1/m)||Xw-y||² is (1/m)·Xᵀe — matching the notebook (no factor of 2)
    fn fit(&mut self, x: &[f64], y: &[f64], m: usize, n: usize) {
        self.feature_mean = column_means(x, m, n);
        self.feature_std  = column_stds(x, m, n, &self.feature_mean);

        let x_norm = normalize(x, m, n, &self.feature_mean, &self.feature_std);
        let x_aug  = augment(&x_norm, m, n);
        let n_aug  = n + 1;

        self.weights = vec![0.0; n_aug];

        let scale = 1.0 / m as f64;

        for _ in 0..self.epochs {
            let preds = mat_vec_mul(&x_aug, &self.weights, m, n_aug);

            let residuals: Vec<f64> = preds.iter()
                .zip(y.iter())
                .map(|(p, t)| p - t)
                .collect();

            let mut grad = mat_t_vec_mul(&x_aug, &residuals, m, n_aug);
            for g in grad.iter_mut() { *g *= scale; }

            match &self.regularization {
                Regularization::L2(lambda) => {
                    for j in 1..n_aug {
                        grad[j] += 2.0 * lambda * self.weights[j];
                    }
                }
                Regularization::L1(lambda) => {
                    for j in 1..n_aug {
                        grad[j] += lambda * self.weights[j].signum();
                    }
                }
                Regularization::None => {}
            }

            for (w, g) in self.weights.iter_mut().zip(grad.iter()) {
                *w -= self.learning_rate * g;
            }
        }
    }

    fn predict(&self, x: &[f64], m: usize, n: usize) -> Vec<f64> {
        let x_norm = normalize(x, m, n, &self.feature_mean, &self.feature_std);
        let x_aug  = augment(&x_norm, m, n);
        mat_vec_mul(&x_aug, &self.weights, m, n + 1)
    }
}