// ─── Trait ───────────────────────────────────────────────────────────────────

// Any supervised model must implement these three methods.
// k_fold_cv and grid_search work against this trait — they don't know or care
// which model is underneath.
pub trait SupervisedModel {
    fn fit(&mut self, x: &[f64], y: &[f64], m: usize, n: usize);
    fn predict(&self, x: &[f64], m: usize, n: usize) -> Vec<f64>;

    fn r2(&self, x: &[f64], y: &[f64], m: usize, n: usize) -> f64 {
        r2(y, &self.predict(x, m, n))
    }

    fn rmse(&self, x: &[f64], y: &[f64], m: usize, n: usize) -> f64 {
        rmse(y, &self.predict(x, m, n))
    }
}

// ─── Regularization ──────────────────────────────────────────────────────────

// Shared by any model that supports regularization.
#[derive(Clone)]
pub enum Regularization {
    None,
    L1(f64),
    L2(f64),
}

// ─── Cross-validation ────────────────────────────────────────────────────────

// K-fold cross-validation, generic over any SupervisedModel.
//
// model_factory is a closure that produces a fresh model for each fold.
// The caller decides how to configure it — this function doesn't care.
//
// Example call:
//   k_fold_cv(x, y, m, n, 5, || LinearRegression::new(0.01, Regularization::L2(0.1), 1000))
pub fn k_fold_cv<M, F, S>(
    x: &[f64], y: &[f64],
    m: usize, n: usize,
    k: usize,
    model_factory: F,
    metric: S,
) -> f64
where
    M: SupervisedModel,
    F: Fn() -> M,   // Fn() means: a closure that takes no args and returns M
    S: Fn(&M, &[f64], &[f64], usize, usize) -> f64,
{
    let fold_size = m / k;
    let mut total_score = 0.0;

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end   = (fold + 1) * fold_size;

        let mut x_train = Vec::new();
        let mut y_train = Vec::new();
        let mut x_test  = Vec::new();
        let mut y_test  = Vec::new();

        for i in 0..m {
            let row = &x[i * n..(i + 1) * n];
            if i >= test_start && i < test_end {
                x_test.extend_from_slice(row);
                y_test.push(y[i]);
            } else {
                x_train.extend_from_slice(row);
                y_train.push(y[i]);
            }
        }

        let m_train = y_train.len();
        let m_test  = y_test.len();

        // model_factory() gives us a brand new model configured by the caller
        // notebook uses RMSE as the validation metric for fold scoring
        let mut model = model_factory();
        model.fit(&x_train, &y_train, m_train, n);
        total_score += metric(&model, &x_test, &y_test, m_test, n);
    }

    total_score / k as f64
}

// Grid search over lambda values using K-fold CV.
//
// model_factory now takes a lambda and returns a configured model.
// Returns all (lambda, rmse) pairs — notebook selects best lambda by lowest avg validation RMSE.
//
// Example call:
//   grid_search(x, y, m, n, 5, |lambda| LinearRegression::new(0.01, Regularization::L2(lambda), 1000), &lambdas)
pub fn grid_search<M, F, S>(
    x: &[f64], y: &[f64],
    m: usize, n: usize,
    k: usize,
    model_factory: F,
    lambdas: &[f64],
    metric: S,
) -> Vec<(f64, f64)>
where
    M: SupervisedModel,
    F: Fn(f64) -> M,
    S: Fn(&M, &[f64], &[f64], usize, usize) -> f64 + Copy,
{
    lambdas.iter()
        .map(|&lambda| {
            let score = k_fold_cv(x, y, m, n, k, || model_factory(lambda), metric);
            (lambda, score)
        })
        .collect()
}

// R² (coefficient of determination) — measures how much of the variance in the target
// the model explains. R²=1 means perfect prediction; R²=0 means the model does no
// better than predicting the mean; negative values mean it's worse than the mean.
pub fn r2(actual: &[f64], predicted: &[f64]) -> f64 {
    // Mean of the actual values — this is what a baseline "predict the mean" model uses
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;

    // SS_tot: total variance in the data — how spread out the actual values are
    let ss_tot: f64 = actual.iter().map(|y| (y - mean).powi(2)).sum();

    // SS_res: residual sum of squares — how much variance our model *fails* to explain
    let ss_res: f64 = actual.iter().zip(predicted).map(|(y, yh)| (y - yh).powi(2)).sum();

    // 1 - (unexplained / total): fraction of variance the model captures
    1.0 - ss_res / ss_tot
}

// RMSE (root mean squared error) — average prediction error in the same units as the target.
// Easier to interpret than MSE: if rings is the target, RMSE is in rings, not rings².
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    // Mean squared error first — average of squared residuals
    let mse: f64 = actual.iter().zip(predicted).map(|(y, yh)| (y - yh).powi(2)).sum::<f64>()
        / actual.len() as f64;

    // Square root brings the error back to the original unit scale
    mse.sqrt()
}