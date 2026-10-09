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
    /// First sample of the IV sums (the filters still start at rest at `k = 0`): leaves out a
    /// transient the model does not explain, e.g. that of an input offset not in the recorded
    /// input (a step at `k = 0`, decaying with the prefilter and the plant's slowest mode), or of
    /// an initial state not at rest. E.g. one period of a periodic input.
    pub evaluated_from: usize,
}

impl<T: Float> Default for SrivcOptions<T> {
    fn default() -> Self {
        Self {
            input_delay: 0,
            input_intersample: InterSample::ZeroOrderHold,
            max_iterations: 20,
            tolerance: T::from(1e-8).unwrap(),
            evaluated_from: 0,
        }
    }
}

/// Result of `identify`.
#[derive(Clone, Debug)]
pub struct SrivcResult<T> {
    /// Identified `e^(-nk ts s) B(s) / (s^q A(s))` (`A` monic; `tf` = `B / (s^q A)`, `delay` = `nk ts`;
    /// `q` the integrators of the prefilter, 0 by `identify`).
    pub model: TransferFunctionWithDelay<T>,
    /// Parameters `θ = [a_1, ..., a_n, b_0, ..., b_m]` of `A(s) = s^n + a_1 s^(n-1) + ... + a_n`, `B(s) = b_0 s^m + ... + b_m`
    /// (`s^q` not included).
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
    #[error("invalid orders: denominator {denominator}, numerator {numerator} (need 1 <= n, m <= n)")]
    InvalidOrder { denominator: usize, numerator: usize },
    #[error("initial model does not match the orders (denominator = {denominator}, numerator <= {numerator})")]
    InitialModel { denominator: usize, numerator: usize },
    #[error("singular normal equations")]
    Singular,
    #[error("invalid prefilter (need a positive, finite cutoff)")]
    InvalidPrefilter,
    #[error("{integrators} integrators need a prefilter")]
    NoPrefilter { integrators: usize },
    #[error("too few samples: {samples}, evaluated from {evaluated_from}, for {parameters} parameters")]
    TooFewSamples { samples: usize, evaluated_from: usize, parameters: usize },
}

/// SRIVC identification of `G(s) = e^(-nk ts s) B(s) / A(s)` (`deg A = n = denominator_order`,
/// `deg B = m = numerator_order`, `A` monic) from the input / output samples `u[k] = u(k ts)`,
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
///
/// Argument order: `denominator_order` (`n`) before `numerator_order` (`m`), as in the notation
/// `(n, m, q, nk)`; e.g. `b_0 / (s^2 + a_1 s + a_2)` is `(2, 0)`.
pub fn identify<T>(
    u: &[T],
    y: &[T],
    ts: T,
    denominator_order: usize,
    numerator_order: usize,
    initialization: &Initialization<T>,
    options: &SrivcOptions<T>,
) -> Result<SrivcResult<T>, SrivcError>
where
    T: Float + AddAssign + MulAssign + ComplexField + RealField,
{
    identify_with_prefilter(u, y, ts, denominator_order, numerator_order, &Prefilter::none(), initialization, options)
}

/// Prefilter of `identify_with_prefilter` for a plant `G(s)` with `q = integrators` poles at, or
/// very close to, the origin (rigid-body mode: `q = 1` for torque -> velocity, `q = 2` for
/// torque -> position): with the order `k = q + 1` and `ω_c = cutoff` \[rad/s\],
///
/// ```text
/// w = s / (s + ω_c)^k u      (pseudo-integral of the input: ≈ u / s^q above ω_c)
/// z = s^k / (s + ω_c)^k y    (high-pass of the output)
/// ```
///
/// No pure integrator is applied to the data, and the input path keeps a zero at the origin, so
/// an input offset or the noise does not drift into `w`, nor a polynomial drift of the output of
/// degree `<= q` (e.g. the integrated unknown input offset, `~ t^q`) into `z`. Put `ω_c` below the
/// lowest excited frequency (e.g. 1/5 .. 1/3 of it), with its transient, `~ k / ω_c`, well within
/// the record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prefilter<T> {
    order: usize,
    cutoff: T,
    integrators: usize,
}

impl<T: Float> Prefilter<T> {
    /// Prefilter for `integrators` (`q`) of cutoff `cutoff` (`ω_c` \[rad/s\], positive and finite),
    /// of order `q + 1`.
    pub fn new(integrators: usize, cutoff: T) -> Result<Self, SrivcError> {
        if !(cutoff > T::zero() && cutoff.is_finite()) {
            return Err(SrivcError::InvalidPrefilter);
        }
        Ok(Self { order: integrators + 1, cutoff, integrators })
    }

    /// No prefilter (`identify`).
    fn none() -> Self {
        Self { order: 0, cutoff: T::zero(), integrators: 0 }
    }

    /// Order `k = q + 1`.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Cutoff `ω_c` \[rad/s\].
    pub fn cutoff(&self) -> T {
        self.cutoff
    }

    /// Number of integrators `q`.
    pub fn integrators(&self) -> usize {
        self.integrators
    }
}

/// SRIVC identification of `R(s) = s^q G(s) = e^(-nk ts s) B(s) / A(s)` of a plant `G(s)` with `q`
/// poles at, or very close to, the origin (`q` = `prefilter.integrators()`), from the
/// pseudo-integrated input `w` and the high-passed output `z` of `prefilter` (see `Prefilter`):
///
/// ```text
/// A(s) z = B(s) w                  (z = L(s) G(s) u = R(s) w with L = s^k / (s + ω_c)^k)
/// ```
///
/// The relation is exact for any `ω_c` (the same filter `L` on both sides, zero initial states),
/// which only weights the data. The prefilter is folded into the SRIVC filter,
/// `s^i / (Â(s) (s + ω_c)^k)`, so the input stays exact for its intersample behaviour; the output,
/// filtered as linear between samples, is not: the error of that interpolation leaves a bias that
/// grows like `k ω_c ts^2` (without the prefilter it cancels at convergence).
///
/// The poles of `G` near the origin are not identified (`deg A = n` = `denominator_order` is the
/// order of `R`, `q` less than that of `G`): the data of a band-limited, finite-length test hold
/// no information on them. The model returned is `G(s) = e^(-nk ts s) R(s) / s^q` (exact poles at
/// the origin, valid in the excited band; replace `1 / s^q` by e.g. `1 / (s + p)` from a separate
/// test of the rigid-body mode if needed), the parameters those of `R`; `R(0) = b_m / a_n` is the
/// gain of the rigid-body mode (`1 / J` for torque -> velocity).
///
/// An input offset not in the recorded input (integrated by the plant into a drift of the output)
/// is removed by the prefilter but for its transient from `k = 0`, which biases the estimate in
/// proportion to the offset; `options.evaluated_from` (e.g. one period) leaves it out.
#[allow(clippy::too_many_arguments)]
pub fn identify_with_prefilter<T>(
    u: &[T],
    y: &[T],
    ts: T,
    denominator_order: usize,
    numerator_order: usize,
    prefilter: &Prefilter<T>,
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
        return Err(SrivcError::InvalidOrder { denominator: n, numerator: m });
    }
    if u.len() < options.evaluated_from + n + m + 1 {
        return Err(SrivcError::TooFewSamples { samples: u.len(), evaluated_from: options.evaluated_from, parameters: n + m + 1 });
    }

    let nk = options.input_delay;
    let u: Vec<T> = (0..u.len()).map(|k| if k >= nk { u[k - nk] } else { T::zero() }).collect();

    let mut theta = match initialization {
        Initialization::StateVariableFilter(lambda) => {
            // (s + λ)^n
            Step::new(&times_power(&[], *lambda, n), prefilter, ts, &u, y, m, options).solve(None)?
        }
        Initialization::Model(tf) => initial_parameter(tf, m, n)?,
    };

    let mut iterations = 0;
    let mut converged = false;
    while iterations < options.max_iterations {
        let a = stabilized(&theta.as_slice()[..n]);
        let b = &theta.as_slice()[n..];
        let next = Step::new(&a, prefilter, ts, &u, y, m, options).solve(Some(b))?;
        iterations += 1;

        let change = (&next - &theta).norm() / next.norm();
        theta = next;
        if change <= options.tolerance {
            converged = true;
            break;
        }
    }

    // s^q A(s)
    let mut denom = vec![T::one()];
    denom.extend_from_slice(&theta.as_slice()[..n]);
    denom.resize(n + 1 + prefilter.integrators, T::zero());
    let tf = TransferFunction::from_polynomials(Polynomial(theta.as_slice()[n..].to_vec()), Polynomial(denom));
    let model = TransferFunctionWithDelay::new(tf, ts * T::from(nk).unwrap());
    Ok(SrivcResult { model, parameter: theta, iterations, converged })
}

/// `θ` of an initial model: `A` made monic, `B` padded to degree `m`.
fn initial_parameter<T: Float + ComplexField>(tf: &TransferFunction<T, Continuous>, m: usize, n: usize) -> Result<DVector<T>, SrivcError> {
    let trim = |p: &Polynomial<T>| p.iter().copied().skip_while(|c| c.is_zero()).collect::<Vec<T>>();
    let (numer, denom) = (trim(&tf.numerator), trim(&tf.denominator));
    if denom.len() != n + 1 || numer.len() > m + 1 {
        return Err(SrivcError::InitialModel { denominator: n, numerator: m });
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

/// Roots `s'` of the monic `A(s)` (`a = [a_1, ..., a_n]`) in units of its root radius `ρ`
/// (`s = ρ s'`), with `ρ`: the roots of `A(ρ s') / ρ^n`, whose coefficients `a_i / ρ^i` are O(1).
/// `None` if they are not found.
fn scaled_roots<T: Float + ComplexField>(a: &[T]) -> Option<(T, Vec<Complex<T>>)> {
    let rho = root_radius(a);
    let mut coefficients = vec![Complex::from(T::one())];
    coefficients.extend(a.iter().enumerate().map(|(i, &c)| Complex::from(c / Float::powi(rho, i as i32 + 1))));
    dka_method(&Polynomial(coefficients)).map(|roots| (rho, roots))
}

/// Whether the monic `A(s)` (`a = [a_1, ..., a_n]`) has a root with `Re s > rate` \[1/s\] (and
/// beyond the rounding, `Re s > ρ sqrt(eps)` with the root radius `ρ` of `scaled_roots`).
/// `false` if the roots are not found.
pub(super) fn has_unstable_root<T: Float + ComplexField>(a: &[T], rate: T) -> bool {
    if a.is_empty() {
        return false;
    }
    scaled_roots(a).is_some_and(|(rho, roots)| {
        let threshold = Float::max(rate / rho, Float::sqrt(T::epsilon()));
        roots.iter().any(|r| r.re > threshold)
    })
}

/// `[a_1, ..., a_n]` of the monic `A(s)` with the roots in the right half-plane reflected to the
/// left half-plane (unchanged if stable, or if the roots are not found).
fn stabilized<T: Float + ComplexField>(a: &[T]) -> Vec<T> {
    let Some((rho, roots)) = scaled_roots(a) else {
        return a.to_vec();
    };
    if roots.iter().all(|r| r.re <= T::zero()) {
        return a.to_vec();
    }
    let reflected: Vec<Complex<T>> = roots.iter().map(|r| Complex::new(-Float::abs(r.re), r.im)).collect();
    vieta_formula(&reflected).0.iter().skip(1).enumerate().map(|(i, c)| c.re * Float::powi(rho, i as i32 + 1)).collect()
}

/// `[c_1, ..., c_(n+count)]` of the monic `A(s) (s + root)^count`, `a = [a_1, ..., a_n]`.
fn times_power<T: Float>(a: &[T], root: T, count: usize) -> Vec<T> {
    let mut c = vec![T::one()];
    c.extend_from_slice(a);
    for _ in 0..count {
        c.push(T::zero());
        for i in (1..c.len()).rev() {
            c[i] = c[i] + root * c[i - 1];
        }
    }
    c.split_off(1)
}

/// One estimation step with the filter `1 / (A(s) P(s))`, `P(s) = (s + ω_c)^k` the prefilter
/// (`P = 1` without it).
struct Step<T> {
    filter: StateVariableFilter<T>,
    /// `1 / A(s)` for the instruments, if the prefilter is on (else `filter`).
    instrument_filter: Option<StateVariableFilter<T>>,
    uf: DMatrix<T>,
    yf: DMatrix<T>,
    n: usize,
    m: usize,
    /// Columns of `uf` / `yf` where `w^(0)` / `z^(0)` start (`k - integrators` / `k`).
    u_shift: usize,
    y_shift: usize,
    /// First sample of the sums.
    start: usize,
}

impl<T: Float + AddAssign + MulAssign + ComplexField + RealField> Step<T> {
    #[allow(clippy::too_many_arguments)]
    fn new(a: &[T], prefilter: &Prefilter<T>, ts: T, u: &[T], y: &[T], m: usize, options: &SrivcOptions<T>) -> Self {
        let filter = StateVariableFilter::new(&times_power(a, prefilter.cutoff, prefilter.order), ts);
        let instrument_filter = (prefilter.order > 0).then(|| StateVariableFilter::new(a, ts));
        let uf = filter.apply(u, options.input_intersample);
        let yf = filter.apply(y, InterSample::FirstOrderHold);
        let (u_shift, y_shift) = (prefilter.order - prefilter.integrators, prefilter.order);
        Self { filter, instrument_filter, uf, yf, n: a.len(), m, u_shift, y_shift, start: options.evaluated_from }
    }

    /// Regressors `φ[k]ᵀ = [-v^(n-1), ..., -v, w^(m), ..., w]` at every sample from `start` on as
    /// the rows of an `(N - start) × (n + m + 1)` matrix, from the filtered `vf` (`v^(i)` in its
    /// column `shift + i`) and the filtered input, by column copies.
    fn regressors(&self, vf: &DMatrix<T>, shift: usize) -> DMatrix<T> {
        let (n, m) = (self.n, self.m);
        let rows = vf.nrows() - self.start;
        let mut regressors = DMatrix::zeros(rows, n + m + 1);
        for i in 0..n {
            let mut column = regressors.column_mut(i);
            column.copy_from(&vf.column(shift + n - 1 - i).rows(self.start, rows));
            column.neg_mut();
        }
        for j in 0..=m {
            regressors.column_mut(n + j).copy_from(&self.uf.column(self.u_shift + m - j).rows(self.start, rows));
        }
        regressors
    }

    /// IV estimate with the auxiliary model `b / A(s)` (least squares if `None`).
    fn solve(&self, b: Option<&[T]>) -> Result<DVector<T>, SrivcError> {
        let n = self.n;
        let xf = b.map(|b| {
            // x̂ = B̂(s) / A(s) w = Σ b_j w_f^(m-j)
            let x_hat: Vec<T> = (0..self.uf.nrows())
                .map(|k| (0..=self.m).fold(T::zero(), |acc, j| acc + b[j] * self.uf[(k, self.u_shift + self.m - j)]))
                .collect();
            self.instrument_filter.as_ref().unwrap_or(&self.filter).apply(&x_hat, InterSample::FirstOrderHold)
        });
        let instrument = xf.as_ref().unwrap_or(&self.yf);

        let size = n + self.m + 1;
        // Σ_k ζ[k] φ[k]ᵀ = Zᵀ Φ and Σ_k ζ[k] y_f^(n)[k] = Zᵀ y_f^(n), with the regressors as rows
        // (one matrix product each instead of an outer product per sample)
        let phi = self.regressors(&self.yf, self.y_shift);
        let zeta_instrument = xf.as_ref().map(|_| self.regressors(instrument, 0));
        let zeta = zeta_instrument.as_ref().unwrap_or(&phi); // least squares: ζ = φ
        let zeta_phi_sum = zeta.tr_mul(&phi);
        let zeta_y_sum: DVector<T> = zeta.tr_mul(&self.yf.column(self.y_shift + n).rows(self.start, self.yf.nrows() - self.start));
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
