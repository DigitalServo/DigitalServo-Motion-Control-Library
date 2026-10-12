//! Output-error (OE) identification by the prediction-error method.

use std::ops::{AddAssign, MulAssign};

use nalgebra::{ComplexField, DMatrix, DVector};
use num_traits::Float;
use thiserror::Error;

use crate::system_identification::arx::Arx;
use crate::system_identification::iv;

/// Starting point of the iterations of `identify`.
#[derive(Clone, Debug)]
pub enum Initialization<T> {
    /// Least squares followed by the given number of IV iterations (`iv::arx::identify`).
    InstrumentalVariable(usize),
    /// Initial model with the orders and input delay of the structure; `F` must be stable.
    Model(Arx<T>),
}

/// Options of `identify`.
#[derive(Clone, Debug)]
pub struct OeOptions<T> {
    /// Maximum number of Gauss-Newton iterations (accepted steps).
    pub max_iterations: usize,
    /// Convergence threshold of the relative parameter change `‖Δθ‖ / ‖θ‖`.
    pub tolerance: T,
    /// First sample of the loss (the simulation still starts at rest at `k = 0`): leaves out a
    /// transient the model does not explain, e.g. that of an initial state not at rest.
    pub evaluated_from: usize,
}

impl<T: Float> Default for OeOptions<T> {
    fn default() -> Self {
        Self { max_iterations: 50, tolerance: T::from(1e-8).unwrap(), evaluated_from: 0 }
    }
}

/// Result of `identify`.
#[derive(Clone, Debug)]
pub struct OeResult<T> {
    /// Identified model: `parameter = [f_1, ..., f_na, b_0, ..., b_nb]`, `transfer_function()`
    /// gives `G(z) = z^-nk B(z) / F(z)`. Its history is cleared.
    pub model: Arx<T>,
    /// Mean squared simulation error `Σ (y - ŷ)^2 / N` over the evaluated samples.
    pub loss: T,
    /// Number of Gauss-Newton iterations done.
    pub iterations: usize,
    /// Whether the relative parameter change fell below `tolerance` (or no step could decrease the
    /// loss any further).
    pub converged: bool,
}

/// Errors of `identify`.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OeError {
    #[error("input and output lengths differ: {input} vs {output}")]
    LengthMismatch { input: usize, output: usize },
    #[error("too few samples: {samples}, evaluated from {evaluated_from}, for {parameters} parameters")]
    TooFewSamples { samples: usize, evaluated_from: usize, parameters: usize },
    #[error("initial model does not match the structure (na, nb, nk) = ({state_order}, {input_order}, {input_delay})")]
    InitialModel { state_order: usize, input_order: usize, input_delay: usize },
    #[error("initial model is unstable")]
    UnstableInitialModel,
    #[error("singular normal equations")]
    Singular,
}

/// Prediction-error identification of the output-error model
///
/// ```text
/// y[k] = z^-nk B(z) / F(z) u[k] + e[k]
/// ŷ[k] = Σ_{i=1..na} f_i ŷ[k-i] + Σ_{i=0..nb} b_i u[k-nk-i]
/// ```
///
/// (`F(z) = 1 - f_1 z^-1 - ... - f_na z^-na`, `B(z) = b_0 + ... + b_nb z^-nb`; the orders and the
/// input delay are those of `structure`, whose parameters and history are not used) from the input /
/// output series of a system at rest at `k = 0` with white measurement noise `e` on the output.
/// Unlike least squares (biased by the noise) and IV, it minimizes the simulation error
///
/// ```text
/// V(θ) = 1/N Σ ε[k]^2,  ε[k] = y[k] - ŷ[k, θ],  θ = [f_1, ..., f_na, b_0, ..., b_nb]
/// ```
///
/// by Levenberg-Marquardt iterations from `initialization`, with the gradient `ψ = ∂ŷ/∂θ` given by
/// the regressor `φ̂[k] = [ŷ[k-1], ..., ŷ[k-na], u[k-nk], ..., u[k-nk-nb]]` filtered by `1 / F`:
///
/// ```text
/// ψ[k] = φ̂[k] + Σ_{i=1..na} f_i ψ[k-i]
/// (H + μ diag H) Δθ = Σ ψ ε,  H = Σ ψ ψᵀ
/// ```
///
/// A step is accepted if `F` stays stable and `V` decreases (then `μ /= 10`), otherwise `μ *= 10`
/// and the step is retried; the iterations stop when `‖Δθ‖ <= tolerance ‖θ‖` or `μ > 1e10`.
pub fn identify<T>(
    u: &[T],
    y: &[T],
    structure: &Arx<T>,
    initialization: &Initialization<T>,
    options: &OeOptions<T>,
) -> Result<OeResult<T>, OeError>
where
    T: Float + AddAssign + MulAssign + ComplexField,
{
    if u.len() != y.len() {
        return Err(OeError::LengthMismatch { input: u.len(), output: y.len() });
    }
    let parameters = structure.parameter_len();
    if u.len() < options.evaluated_from + parameters {
        return Err(OeError::TooFewSamples { samples: u.len(), evaluated_from: options.evaluated_from, parameters });
    }

    let mut model = match initialization {
        Initialization::InstrumentalVariable(iterations) => {
            iv::arx::identify(u, y, structure, *iterations).ok_or(OeError::Singular)?
        }
        Initialization::Model(model) => {
            let (na, nb, nk) = (structure.state_order(), structure.input_order(), structure.input_delay());
            if (model.state_order(), model.input_order(), model.input_delay()) != (na, nb, nk) {
                return Err(OeError::InitialModel { state_order: na, input_order: nb, input_delay: nk });
            }
            model.clone()
        }
    };
    model.clear_history();
    if !is_stable(&model) {
        return Err(OeError::UnstableInitialModel);
    }

    let start = options.evaluated_from;
    let mut loss = simulation_loss(u, y, &model, start);
    let mut mu = T::from(1e-3).unwrap();
    let mu_max = T::from(1e10).unwrap();
    let mu_min = T::from(1e-12).unwrap();
    let mut iterations = 0;
    let mut converged = false;

    'outer: while iterations < options.max_iterations {
        let (h, g) = normal_equations(u, y, &model, start);
        loop {
            let mut damped = h.clone();
            for i in 0..parameters {
                damped[(i, i)] += mu * h[(i, i)].max(T::epsilon());
            }
            let delta = damped.lu().solve(&g).ok_or(OeError::Singular)?;

            let mut trial = model.clone();
            trial.parameter += &delta;
            let trial_loss = if is_stable(&trial) { simulation_loss(u, y, &trial, start) } else { T::infinity() };
            if trial_loss < loss {
                model = trial;
                loss = trial_loss;
                mu = (mu / T::from(10).unwrap()).max(mu_min);
                iterations += 1;
                let tolerance2 = options.tolerance * options.tolerance;
                if delta.dot(&delta) <= tolerance2 * model.parameter.dot(&model.parameter) {
                    converged = true;
                    break 'outer;
                }
                break;
            }
            mu *= T::from(10).unwrap();
            if mu > mu_max {
                converged = true;
                break 'outer;
            }
        }
    }

    Ok(OeResult { model, loss, iterations, converged })
}

/// Mean squared simulation error `Σ_{k >= start} (y - ŷ)^2 / (N - start)` of `model` (at rest at `k = 0`).
fn simulation_loss<T: Float + AddAssign + ComplexField>(u: &[T], y: &[T], model: &Arx<T>, start: usize) -> T {
    let mut sim = model.clone();
    sim.clear_history();
    let mut y_hat = T::zero();
    let mut sum = T::zero();
    for (k, (&uk, &yk)) in u.iter().zip(y).enumerate() {
        sim.push(uk, y_hat);
        y_hat = sim.predict();
        if k >= start {
            sum += (yk - y_hat) * (yk - y_hat);
        }
    }
    let loss = sum / T::from(u.len() - start).unwrap();
    if Float::is_finite(loss) { loss } else { T::infinity() }
}

/// `H = Σ ψ ψᵀ` and `g = Σ ψ ε` over `k >= start` (see `identify`).
fn normal_equations<T: Float + AddAssign + MulAssign + ComplexField>(
    u: &[T],
    y: &[T],
    model: &Arx<T>,
    start: usize,
) -> (DMatrix<T>, DVector<T>) {
    let n = model.parameter_len();
    let na = model.state_order();
    let f = model.parameter.rows(0, na).clone_owned();

    let mut sim = model.clone();
    sim.clear_history();
    let mut y_hat = T::zero();
    // ψ[k-1], ..., ψ[k-na]
    let mut psi_history: Vec<DVector<T>> = vec![DVector::zeros(n); na];
    let mut h = DMatrix::zeros(n, n);
    let mut g = DVector::zeros(n);
    for (k, (&uk, &yk)) in u.iter().zip(y).enumerate() {
        sim.push(uk, y_hat);
        let phi = sim.regressor();
        y_hat = phi.dot(&sim.parameter);

        let mut psi = phi;
        for (fi, psi_i) in f.iter().zip(&psi_history) {
            psi.axpy(*fi, psi_i, T::one());
        }
        if k >= start {
            h += &psi * psi.transpose();
            g += &psi * (yk - y_hat);
        }
        if na > 0 {
            psi_history.rotate_right(1);
            psi_history[0] = psi;
        }
    }
    (h, g)
}

/// Whether the roots of `F(z) = 1 - f_1 z^-1 - ... - f_na z^-na` are inside the unit circle
/// (Schur-Cohn step-down: every reflection coefficient `|k| < 1`).
fn is_stable<T: Float + ComplexField>(model: &Arx<T>) -> bool {
    let na = model.state_order();
    let mut a: Vec<T> = std::iter::once(T::one()).chain(model.parameter.iter().take(na).map(|&f| -f)).collect();
    if a.iter().any(|v| !Float::is_finite(*v)) {
        return false;
    }
    while a.len() > 1 {
        let n = a.len() - 1;
        let k = a[n] / a[0];
        if Float::abs(k) >= T::one() {
            return false;
        }
        let next: Vec<T> = (0..n).map(|i| (a[i] - k * a[n - i]) / (T::one() - k * k)).collect();
        a = next;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schur_cohn_stability() {
        let model = |f: &[f64]| {
            let mut m = Arx::<f64>::new(f.len(), 0);
            m.parameter.rows_mut(0, f.len()).copy_from_slice(f);
            m
        };
        // (1 - 0.9 z^-1)(1 - 0.8 z^-1), complex pair of radius √0.7, and z = 1.1, z = -1
        assert!(is_stable(&model(&[1.7, -0.72])));
        assert!(is_stable(&model(&[1.5, -0.7])));
        assert!(!is_stable(&model(&[1.1])));
        assert!(!is_stable(&model(&[-1.0])));
        assert!(!is_stable(&model(&[1.9, -0.88]))); // poles 1.1, 0.8
        assert!(is_stable(&model(&[])));
    }
}
