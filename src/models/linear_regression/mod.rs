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
    pub loss_history: Vec<f64>,
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
            loss_history: Vec::new(),
        }
    }

}

impl SupervisedModel for LinearRegression {

    // w̃ ← w̃ − α · [(2/m) X̃ᵀ(X̃w̃ − y) + regularization_term]
    fn fit(&mut self, x: &[f64], y: &[f64], m: usize, n: usize) {
        self.feature_mean = column_means(x, m, n);
        self.feature_std  = column_stds(x, m, n, &self.feature_mean);

        let x_norm = normalize(x, m, n, &self.feature_mean, &self.feature_std);
        let x_aug  = augment(&x_norm, m, n);
        let n_aug  = n + 1;

        self.weights = vec![0.0; n_aug];
        self.loss_history = Vec::with_capacity(self.epochs);

        let scale = 2.0 / m as f64;

        for _ in 0..self.epochs {
            let preds = mat_vec_mul(&x_aug, &self.weights, m, n_aug);

            let residuals: Vec<f64> = preds.iter()
                .zip(y.iter())
                .map(|(p, t)| p - t)
                .collect();

            let loss = residuals.iter().map(|e| e * e).sum::<f64>() / m as f64;
            self.loss_history.push(loss);

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

   fn mse(&self, x: &[f64], y: &[f64], m: usize, n: usize) -> f64 {
        let preds = self.predict(x, m, n);
        preds.iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / m as f64
    }
}