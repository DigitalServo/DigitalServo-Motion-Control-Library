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
    /// Least squares (ARX, `iv::arx::identify` with no IV iteration). An unstable estimate is
    /// stabilized (`Arx::stabilized`).
    ///
    /// Usable only with little noise on the output and a sampling not much faster than the plant:
    /// least squares minimizes the equation error `F(q) y - B(q) u`, i.e. the output error weighted
    /// by `|F(e^jω)|^2`, which emphasizes the high frequencies when the poles of `F` are near `z = 1`,
    /// and the noise biases it further (a lightly damped pair often comes out as two real poles).
    /// Otherwise the iterations end in a local minimum, e.g. for `ω0 / 2π = 20 Hz`, `ζ = 0.02`:
    ///
    /// ```text
    /// ts      noise (std / std of y)   V of OE from least squares   from SRIV
    /// 1 ms    2 %                      ≈ σ² (14 .. 40 iterations)   ≈ σ² (1 iteration)
    /// 1 ms    17 %                     23 .. 31 σ²                  ≈ σ²
    /// 0.1 ms  2 %                      ≈ 3000 σ²                    ≈ σ²
    /// ```
    ///
    /// Kept for comparison; prefer `Sriv`.
    LeastSquares,
    /// Least squares followed by the given number of IV iterations (`iv::arx::identify`). An
    /// unstable estimate is stabilized (`Arx::stabilized`).
    InstrumentalVariable(usize),
    /// Least squares followed by the given number of SRIV iterations (`iv::arx::identify_sriv`):
    /// its fixed point is a stationary point of the loss. Recommended: with the poles of `F` near
    /// `z = 1` (lightly damped plants sampled fast) the plain IV estimate is often unstable or far
    /// from the minimum. An unstable estimate is stabilized (`Arx::stabilized`).
    Sriv(usize),
    /// Initial model with the orders and input delay of the structure; `F` must be stable
    /// (`UnstableInitialModel` otherwise: it is not stabilized).
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
    /// Why the iterations stopped.
    pub termination: Termination,
}

impl<T> OeResult<T> {
    /// Whether the relative parameter change fell below `tolerance` (`Termination::Converged`).
    pub fn converged(&self) -> bool {
        self.termination == Termination::Converged
    }
}

/// Why the iterations of `identify` stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// The relative parameter change of an accepted step fell below `tolerance`.
    Converged,
    /// No step decreased the loss up to `μ > 1e10`: the loss cannot be decreased further within
    /// rounding (e.g. at a minimum reached before `tolerance`), but this is not a guarantee of one.
    Stalled,
    /// `max_iterations` steps were accepted without convergence.
    MaxIterations,
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

/// Prediction-error identification of the output-error (OE) model
///
/// ```text
/// y[k] = z^-nk B(z) / F(z) u[k] + e[k]
/// F(z) = 1 - f_1 z^-1 - ... - f_na z^-na
/// B(z) = b_0 + b_1 z^-1 + ... + b_nb z^-nb
/// ```
///
/// from the input / output series. The orders `na` (often written `nf` for OE models), `nb` and the
/// input delay `nk` are those of `structure`, whose parameters and history are not used.
///
/// Assumptions on the data:
/// - The system is at rest at `k = 0` (`u[k] = ŷ[k] = 0` for `k < 0`). The transient of a nonzero
///   initial state is fitted as model error; `options.evaluated_from` leaves it out of `V`.
/// - Open loop: `e` is uncorrelated with `u`. This alone makes the estimate of `B / F` consistent,
///   even for colored `e`, whereas least squares (ARX) is biased by the noise. If `e` is also white
///   and Gaussian, the estimate is the maximum-likelihood one and asymptotically efficient, whereas
///   IV is consistent but not minimum variance.
///
/// It minimizes the simulation error
///
/// ```text
/// V(θ) = 1/N Σ ε[k]^2,  ε[k] = y[k] - ŷ[k, θ],  θ = [f_1, ..., f_na, b_0, ..., b_nb]
/// ŷ[k, θ] = Σ_{i=1..na} f_i ŷ[k-i] + Σ_{i=0..nb} b_i u[k-nk-i]
/// ```
///
/// which is nonlinear in `θ` since the regressor `φ̂[k] = [ŷ[k-1], ..., ŷ[k-na], u[k-nk], ...,
/// u[k-nk-nb]]` contains the simulated output. The gradient `ψ = ∂ŷ/∂θ` is `φ̂` filtered by `1 / F`,
/// and each step solves the Levenberg-Marquardt equations (Gauss-Newton with Marquardt scaling):
///
/// ```text
/// ψ[k] = φ̂[k] + Σ_{i=1..na} f_i ψ[k-i]
/// (H + μ diag H) Δθ = Σ ψ ε,  H = Σ ψ ψᵀ
/// ```
///
/// A step `θ + Δθ` is accepted if `F` stays stable (needed by both `ŷ` and the filter of `ψ`) and
/// `V` decreases, then `μ /= 10`; otherwise `μ *= 10` and the step is solved again with the same
/// `H` and `Σ ψ ε`. The iterations stop when `‖Δθ‖ <= tolerance ‖θ‖` (`Termination::Converged`),
/// when `μ > 1e10` (no step decreases `V`: `Termination::Stalled`, not a guarantee of a minimum),
/// or after `max_iterations` accepted steps (`Termination::MaxIterations`).
///
/// `V` is non-convex and the iterations reach the local minimum near `initialization`, which should
/// be a consistent estimate with a stable `F` (e.g. `Initialization::Sriv`, whose fixed point is a
/// stationary point of `V`, or `InstrumentalVariable`; `LeastSquares` only with little noise). The
/// least-squares, IV and SRIV estimates are stabilized, which keeps the iterations running but may
/// start them far from the minimum.
///
/// At the minimum, `λ̂ H^-1` with `λ̂ = N V(θ̂) / (N - d)`, `d = na + nb + 1`, estimates the
/// covariance of the parameters only if `e` is white (for colored `e`, a sandwich form is needed).
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
        Initialization::LeastSquares => iv::arx::identify(u, y, structure, 0).ok_or(OeError::Singular)?.stabilized(),
        Initialization::InstrumentalVariable(iterations) => {
            iv::arx::identify(u, y, structure, *iterations).ok_or(OeError::Singular)?.stabilized()
        }
        Initialization::Sriv(iterations) => {
            iv::arx::identify_sriv(u, y, structure, *iterations).ok_or(OeError::Singular)?.stabilized()
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
    if !model.is_stable() {
        return Err(OeError::UnstableInitialModel);
    }

    let start = options.evaluated_from;
    let mut loss = simulation_loss(u, y, &model, start);
    let mut mu = T::from(1e-3).unwrap();
    let mu_max = T::from(1e10).unwrap();
    let mu_min = T::from(1e-12).unwrap();
    let mut iterations = 0;
    let mut termination = Termination::MaxIterations;

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
            let trial_loss = if trial.is_stable() { simulation_loss(u, y, &trial, start) } else { T::infinity() };
            if trial_loss < loss {
                model = trial;
                loss = trial_loss;
                mu = (mu / T::from(10).unwrap()).max(mu_min);
                iterations += 1;
                let tolerance2 = options.tolerance * options.tolerance;
                if delta.dot(&delta) <= tolerance2 * model.parameter.dot(&model.parameter) {
                    termination = Termination::Converged;
                    break 'outer;
                }
                break;
            }
            mu *= T::from(10).unwrap();
            if mu > mu_max {
                termination = Termination::Stalled;
                break 'outer;
            }
        }
    }

    Ok(OeResult { model, loss, iterations, termination })
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
