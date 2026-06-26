use crate::math::{augment, column_means, column_stds, mat_t_vec_mul, mat_vec_mul, normalize};
use crate::ml::Regularization;
use crate::ml::SupervisedModel;

pub struct LogisticRegression {
    pub weights: Vec<f64>,
    pub feature_mean: Vec<f64>,
    pub feature_std: Vec<f64>,
    learning_rate: f64,
    regularization: Regularization,
    pub epochs: usize,
    pub loss_train_history: Vec<f64>,
    pub loss_val_history: Vec<f64>,
    pub acc_train_history: Vec<f64>,
    pub acc_val_history: Vec<f64>,
}

// -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)] — binary cross-entropy loss
// clamp prevents log(0) blowing up to -∞
fn log_loss(y: &[f64], preds: &[f64]) -> f64 {
    let m = y.len() as f64;
    let eps = 1e-15;
    y.iter().zip(preds.iter())
        .map(|(&t, &p)| {
            let p = p.clamp(eps, 1.0 - eps);
            t * p.ln() + (1.0 - t) * (1.0 - p).ln()
        })
        .sum::<f64>() * (-1.0 / m)
}

// fraction of predictions that match the true labels after thresholding at 0.5
fn accuracy(preds: &[f64], y: &[f64]) -> f64 {
    let correct = preds.iter().zip(y.iter())
        .filter(|(&p, &t)| (p >= 0.5) == (t > 0.5))
        .count();
    correct as f64 / y.len() as f64
}

impl LogisticRegression {
    pub fn new(learning_rate: f64, regularization: Regularization, epochs: usize) -> Self {
        Self {
            weights: Vec::new(),
            feature_mean: Vec::new(),
            feature_std: Vec::new(),
            learning_rate,
            regularization,
            epochs,
            loss_train_history: Vec::new(),
            loss_val_history: Vec::new(),
            acc_train_history: Vec::new(),
            acc_val_history: Vec::new(),
        }
    }

    // ŷ = σ(Xw) = 1 / (1 + e^{-Xw}) — apply sigmoid to the linear output to get class probabilities
    fn sigmoid_predict(&self, x_aug: &[f64], m: usize, n_aug: usize) -> Vec<f64> {
        let z = mat_vec_mul(x_aug, &self.weights, m, n_aug);
        z.iter().map(|&zi| 1.0 / (1.0 + (-zi).exp())).collect()
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

        self.weights           = vec![0.0; n_aug];
        self.loss_train_history = Vec::with_capacity(self.epochs);
        self.loss_val_history   = Vec::with_capacity(self.epochs);
        self.acc_train_history  = Vec::with_capacity(self.epochs);
        self.acc_val_history    = Vec::with_capacity(self.epochs);

        // gradient of binary cross-entropy ∇L = (1/m)·Xᵀ(ŷ-y) — same form as linear regression but ŷ = σ(Xw)
        let scale = 1.0 / m_train as f64;

        for _ in 0..self.epochs {
            let preds = self.sigmoid_predict(&x_train_aug, m_train, n_aug);

            // Track loss and accuracy on training set before weight update
            self.loss_train_history.push(log_loss(y_train, &preds));
            self.acc_train_history.push(accuracy(&preds, y_train));

            // Record val metrics using current weights
            let val_preds = self.sigmoid_predict(&x_val_aug, m_val, n_aug);
            self.loss_val_history.push(log_loss(y_val, &val_preds));
            self.acc_val_history.push(accuracy(&val_preds, y_val));

            // ŷ - y: error signal — difference between predicted probabilities and true labels
            let residuals: Vec<f64> = preds.iter()
                .zip(y_train.iter())
                .map(|(p, t)| p - t)
                .collect();

            // ∇L = (1/m) Xᵀ(ŷ - y) — gradient of binary cross-entropy wrt weights
            let mut grad = mat_t_vec_mul(&x_train_aug, &residuals, m_train, n_aug);
            for g in grad.iter_mut() { *g *= scale; }

            match &self.regularization {
                Regularization::L2(lambda) => {
                    // notebook scales regularization by (λ/m), no factor of 2
                    for j in 1..n_aug { grad[j] += (lambda / m_train as f64) * self.weights[j]; }
                }
                Regularization::L1(lambda) => {
                    for j in 1..n_aug { grad[j] += (lambda / m_train as f64) * self.weights[j].signum(); }
                }
                Regularization::None => {}
            }

            for (w, g) in self.weights.iter_mut().zip(grad.iter()) {
                *w -= self.learning_rate * g;
            }
        }
    }
}

impl SupervisedModel for LogisticRegression {

    // w̃ ← w̃ − α · [(1/m) X̃ᵀ(σ(X̃w̃) − y) + regularization_term]
    fn fit(&mut self, x: &[f64], y: &[f64], m: usize, n: usize) {
        self.feature_mean = column_means(x, m, n);
        self.feature_std  = column_stds(x, m, n, &self.feature_mean);

        let x_norm = normalize(x, m, n, &self.feature_mean, &self.feature_std);
        let x_aug  = augment(&x_norm, m, n);
        let n_aug  = n + 1;

        self.weights = vec![0.0; n_aug];

        // gradient of binary cross-entropy ∇L = (1/m)·Xᵀ(ŷ-y) — same form as linear regression but ŷ = σ(Xw)
        let scale = 1.0 / m as f64;

        for _ in 0..self.epochs {
            let preds = self.sigmoid_predict(&x_aug, m, n_aug);

            // ŷ - y: error signal — difference between predicted probabilities and true labels
            let residuals: Vec<f64> = preds.iter()
                .zip(y.iter())
                .map(|(p, t)| p - t)
                .collect();

            // ∇L = (1/m) Xᵀ(ŷ - y) — gradient of binary cross-entropy wrt weights
            let mut grad = mat_t_vec_mul(&x_aug, &residuals, m, n_aug);
            for g in grad.iter_mut() { *g *= scale; }

            match &self.regularization {
                Regularization::L2(lambda) => {
                    // notebook scales regularization by (λ/m), no factor of 2
                    for j in 1..n_aug {
                        grad[j] += (lambda / m as f64) * self.weights[j];
                    }
                }
                Regularization::L1(lambda) => {
                    for j in 1..n_aug {
                        grad[j] += (lambda / m as f64) * self.weights[j].signum();
                    }
                }
                Regularization::None => {}
            }

            for (w, g) in self.weights.iter_mut().zip(grad.iter()) {
                *w -= self.learning_rate * g;
            }
        }
    }

    // ŷ = σ(Xw) — returns class probabilities in [0, 1]; threshold at 0.5 for class labels
    fn predict(&self, x: &[f64], m: usize, n: usize) -> Vec<f64> {
        let x_norm = normalize(x, m, n, &self.feature_mean, &self.feature_std);
        let x_aug  = augment(&x_norm, m, n);
        self.sigmoid_predict(&x_aug, m, n + 1)
    }
}
