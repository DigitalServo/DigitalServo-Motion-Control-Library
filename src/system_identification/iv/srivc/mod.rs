//! Simplified refined instrumental variable method for continuous-time systems (SRIVC).

use std::ops::{AddAssign, MulAssign};

use nalgebra::{ComplexField, DMatrix, DVector, RealField};
use num_complex::Complex;
use num_traits::Float;
use thiserror::Error;

use crate::{Continuous, Polynomial, TransferFunction, TransferFunctionWithDelay, dka_method, vieta_formula};
use crate::discretize::state_variable_filter::{StateVariableFilter, root_radius};

pub use crate::discretize::InterSample;

pub mod search;
pub use search::{search, Candidate, Outcome, SearchOptions, Structure, StructureSearch};

/// Starting point of the SRIVC iterations.
#[derive(Clone, Debug)]
pub enum Initialization<T> {
    /// Least squares with the state-variable filter `1 / (s + λ)^n` of the given `λ` \[rad/s\]
    /// (around the bandwidth of the plant), whose estimate starts the iterations.
    StateVariableFilter(T),
    /// Initial model `B(s) / A(s)` (e.g. a discrete-time estimate converted to continuous time),
    /// with `deg A = n` and `deg B <= m`.
    Model(TransferFunction<T, Continuous>),
}

/// Options of `identify`.
#[derive(Clone, Debug)]
pub struct SrivcOptions<T> {
    /// Input delay `nk` \[samples\]: the model is `e^(-nk ts s) B(s) / A(s)`.
    pub input_delay: usize,
    /// Intersample behaviour of the input (the output is filtered as `FirstOrderHold`).
    pub input_intersample: InterSample,
    /// Maximum number of IV iterations.
    pub max_iterations: usize,
    /// Convergence threshold of the relative parameter change `‖Δθ‖ / ‖θ‖`.
    pub tolerance: T,
}

impl<T: Float> Default for SrivcOptions<T> {
    fn default() -> Self {
        Self {
            input_delay: 0,
            input_intersample: InterSample::ZeroOrderHold,
            max_iterations: 20,
            tolerance: T::from(1e-8).unwrap(),
        }
    }
}

/// Result of `identify`.
#[derive(Clone, Debug)]
pub struct SrivcResult<T> {
    /// Identified `e^(-nk ts s) B(s) / A(s)` (`A` monic; `tf` = `B / A`, `delay` = `nk ts`).
    pub model: TransferFunctionWithDelay<T>,
    /// Parameters `θ = [a_1, ..., a_n, b_0, ..., b_m]` of `A(s) = s^n + a_1 s^(n-1) + ... + a_n`, `B(s) = b_0 s^m + ... + b_m`.
    pub parameter: DVector<T>,
    /// Number of IV iterations done.
    pub iterations: usize,
    /// Whether the relative parameter change fell below `tolerance`.
    pub converged: bool,
}

/// Errors of `identify`.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SrivcError {
    #[error("input and output lengths differ: {input} vs {output}")]
    LengthMismatch { input: usize, output: usize },
    #[error("invalid orders: numerator {numerator}, denominator {denominator} (need 1 <= n, m <= n)")]
    InvalidOrder { numerator: usize, denominator: usize },
    #[error("initial model does not match the orders (numerator <= {numerator}, denominator = {denominator})")]
    InitialModel { numerator: usize, denominator: usize },
    #[error("singular normal equations")]
    Singular,
}

/// SRIVC identification of `G(s) = e^(-nk ts s) B(s) / A(s)` (`deg B = m = numerator_order`,
/// `deg A = n = denominator_order`, `A` monic) from the input / output samples `u[k] = u(k ts)`,
/// `y[k] = y(k ts)` of a system at rest at `k = 0`, with white measurement noise on the output.
///
/// With the current estimate `Â`, `B̂`, every signal is filtered by `s^i / Â(s)` (`i = 0 ..= n`):
///
/// ```text
/// y_f^(n) = -a_1 y_f^(n-1) - ... - a_n y_f + b_0 u_f^(m) + ... + b_m u_f = φ_fᵀ θ
/// θ = (Σ ζ_f φ_fᵀ)^-1 Σ ζ_f y_f^(n)
/// ```
///
/// with the instruments `ζ_f` = `φ_f` with `y` replaced by the noise-free auxiliary model output
/// `x̂ = B̂(s) / Â(s) u`. The new estimate gives the next filter and auxiliary model, until the
/// parameters converge. Unstable poles of `Â` are reflected to the left half-plane for filtering.
/// The filters are discretized exactly for the given intersample behaviour (the output and `x̂` as
/// `FirstOrderHold`), so `ts` should be small against the time constants of the plant.
pub fn identify<T>(
    u: &[T],
    y: &[T],
    ts: T,
    numerator_order: usize,
    denominator_order: usize,
    initialization: &Initialization<T>,
    options: &SrivcOptions<T>,
) -> Result<SrivcResult<T>, SrivcError>
where
    T: Float + AddAssign + MulAssign + ComplexField + RealField,
{
    let (m, n) = (numerator_order, denominator_order);
    if u.len() != y.len() {
        return Err(SrivcError::LengthMismatch { input: u.len(), output: y.len() });
    }
    if n == 0 || m > n {
        return Err(SrivcError::InvalidOrder { numerator: m, denominator: n });
    }

    let nk = options.input_delay;
    let u: Vec<T> = (0..u.len()).map(|k| if k >= nk { u[k - nk] } else { T::zero() }).collect();

    let mut theta = match initialization {
        Initialization::StateVariableFilter(lambda) => {
            // (s + λ)^n
            let mut a = vec![T::one()];
            for _ in 0..n {
                a.push(T::zero());
                for i in (1..a.len()).rev() {
                    a[i] = a[i] + *lambda * a[i - 1];
                }
            }
            Step::new(&a[1..], ts, &u, y, m, options.input_intersample).solve(None)?
        }
        Initialization::Model(tf) => initial_parameter(tf, m, n)?,
    };

    let mut iterations = 0;
    let mut converged = false;
    while iterations < options.max_iterations {
        let a = stabilized(&theta.as_slice()[..n]);
        let b = &theta.as_slice()[n..];
        let next = Step::new(&a, ts, &u, y, m, options.input_intersample).solve(Some(b))?;
        iterations += 1;

        let change = (&next - &theta).norm() / next.norm();
        theta = next;
        if change <= options.tolerance {
            converged = true;
            break;
        }
    }

    let mut denom = vec![T::one()];
    denom.extend_from_slice(&theta.as_slice()[..n]);
    let tf = TransferFunction::from_polynomials(Polynomial(theta.as_slice()[n..].to_vec()), Polynomial(denom));
    let model = TransferFunctionWithDelay::new(tf, ts * T::from(nk).unwrap());
    Ok(SrivcResult { model, parameter: theta, iterations, converged })
}

/// `θ` of an initial model: `A` made monic, `B` padded to degree `m`.
fn initial_parameter<T: Float + ComplexField>(tf: &TransferFunction<T, Continuous>, m: usize, n: usize) -> Result<DVector<T>, SrivcError> {
    let trim = |p: &Polynomial<T>| p.iter().copied().skip_while(|c| c.is_zero()).collect::<Vec<T>>();
    let (numer, denom) = (trim(&tf.numerator), trim(&tf.denominator));
    if denom.len() != n + 1 || numer.len() > m + 1 {
        return Err(SrivcError::InitialModel { numerator: m, denominator: n });
    }
    let lead = denom[0];
    let mut theta = DVector::zeros(n + m + 1);
    for i in 0..n {
        theta[i] = denom[i + 1] / lead;
    }
    for (i, &c) in numer.iter().enumerate() {
        theta[n + m + 1 - numer.len() + i] = c / lead;
    }
    Ok(theta)
}

/// `[a_1, ..., a_n]` of the monic `A(s)` with the roots in the right half-plane reflected to the
/// left half-plane (unchanged if stable, or if the roots are not found).
fn stabilized<T: Float + ComplexField>(a: &[T]) -> Vec<T> {
    // Roots of A(ρ s') / ρ^n, whose coefficients a_i / ρ^i are O(1)
    let rho = root_radius(a);
    let mut coefficients = vec![Complex::from(T::one())];
    coefficients.extend(a.iter().enumerate().map(|(i, &c)| Complex::from(c / Float::powi(rho, i as i32 + 1))));
    let Some(roots) = dka_method(&Polynomial(coefficients)) else {
        return a.to_vec();
    };
    if roots.iter().all(|r| r.re <= T::zero()) {
        return a.to_vec();
    }
    let reflected: Vec<Complex<T>> = roots.iter().map(|r| Complex::new(-Float::abs(r.re), r.im)).collect();
    vieta_formula(&reflected).0.iter().skip(1).enumerate().map(|(i, c)| c.re * Float::powi(rho, i as i32 + 1)).collect()
}

/// One estimation step with the filter `1 / A(s)`.
struct Step<T> {
    filter: StateVariableFilter<T>,
    uf: DMatrix<T>,
    yf: DMatrix<T>,
    m: usize,
}

impl<T: Float + AddAssign + MulAssign + ComplexField + RealField> Step<T> {
    fn new(a: &[T], ts: T, u: &[T], y: &[T], m: usize, input_intersample: InterSample) -> Self {
        let filter = StateVariableFilter::new(a, ts);
        let uf = filter.apply(u, input_intersample);
        let yf = filter.apply(y, InterSample::FirstOrderHold);
        Self { filter, uf, yf, m }
    }

    /// Regressor `[-v^(n-1), ..., -v, u^(m), ..., u]` at sample `k` from the filtered `vf`.
    fn regressor(&self, vf: &DMatrix<T>, k: usize) -> DVector<T> {
        let (n, m) = (self.filter.order(), self.m);
        let mut phi = DVector::zeros(n + m + 1);
        for i in 0..n {
            phi[i] = -vf[(k, n - 1 - i)];
        }
        for j in 0..=m {
            phi[n + j] = self.uf[(k, m - j)];
        }
        phi
    }

    /// IV estimate with the auxiliary model `b / A(s)` (least squares if `None`).
    fn solve(&self, b: Option<&[T]>) -> Result<DVector<T>, SrivcError> {
        let n = self.filter.order();
        let xf = b.map(|b| {
            // x̂ = B̂(s) / A(s) u = Σ b_j u_f^(m-j)
            let x_hat: Vec<T> = (0..self.uf.nrows())
                .map(|k| (0..=self.m).fold(T::zero(), |acc, j| acc + b[j] * self.uf[(k, self.m - j)]))
                .collect();
            self.filter.apply(&x_hat, InterSample::FirstOrderHold)
        });
        let instrument = xf.as_ref().unwrap_or(&self.yf);

        let size = n + self.m + 1;
        let mut zeta_phi_sum = DMatrix::zeros(size, size);
        let mut zeta_y_sum = DVector::zeros(size);
        for k in 0..self.yf.nrows() {
            let phi = self.regressor(&self.yf, k);
            let zeta = self.regressor(instrument, k);
            zeta_y_sum += &zeta * self.yf[(k, n)];
            zeta_phi_sum += &zeta * &phi.transpose();
        }
        // Equilibration: rows and columns scaled to unit max norm
        let col: Vec<T> = (0..size).map(|j| zeta_phi_sum.column(j).amax()).collect();
        let row: Vec<T> = (0..size).map(|i| zeta_phi_sum.row(i).amax()).collect();
        let mut scaled = zeta_phi_sum.clone();
        let mut rhs = zeta_y_sum.clone();
        for i in 0..size {
            for j in 0..size {
                scaled[(i, j)] /= row[i] * col[j];
            }
            rhs[i] /= row[i];
        }
        let x = scaled.lu().solve(&rhs).ok_or(SrivcError::Singular)?;
        Ok(DVector::from_iterator(size, (0..size).map(|j| x[j] / col[j])))
    }
}
