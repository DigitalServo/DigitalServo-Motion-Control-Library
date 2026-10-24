//! Levy's method and the Sanathanan-Koerner iteration.

use std::ops::{AddAssign, MulAssign};

use nalgebra::{Complex, ComplexField, DMatrix, DVector};
use num_traits::{Float, Zero};
use thiserror::Error;

use crate::{Continuous, FrequencyResponse, Polynomial, TransferFunction};

/// Errors of Levy's method and the Sanathanan-Koerner iteration.
#[derive(Error, Debug)]
pub enum LevyIdentificationError {
    /// No samples.
    #[error("No sampled data provided")]
    EmptyData,
    /// A sample whose frequency is not finite and non-negative, or whose response is not finite.
    #[error("Invalid sample {index}: finite ω >= 0 and finite G(jω) needed")]
    InvalidSample { index: usize },
    /// The normal equations are singular.
    #[error("Matrix is singular or numerically unstable (check order and data)")]
    SingularMatrix,
    /// No iteration was run (`max_iter == 0`).
    #[error("Failed to iterate (check data and iterations)")]
    IterationError,
    /// The previous estimate has a different denominator order.
    #[error("Order of previous result does not match current order")]
    OrderMismatch,
    /// The weights are not one per sample, or not all finite and non-negative.
    #[error("Invalid weights: {len} for {samples} samples, all finite and non-negative needed")]
    InvalidWeights { len: usize, samples: usize },
    /// A sample whose response is zero (or so small that `1 / |G|` is not finite), for the
    /// relative error of `sanathanan_koerner_relative`.
    #[error("Zero response at sample {index}: no relative error")]
    ZeroResponse { index: usize },
}

/// Options of the Sanathanan-Koerner iteration.
#[derive(Debug, Clone)]
pub struct SanathananKoernerOptions<T> {
    /// Maximum number of Levy steps (`1` is Levy's method alone).
    pub max_iter: usize,
    /// Convergence tolerance on the relative change of the denominator over the samples,
    /// `max_i |A_k(jω_i) - A_{k-1}(jω_i)| / |A_{k-1}(jω_i)|`: that of the weights `1 / |A|`.
    pub tol: T,
}

impl<T: Float> Default for SanathananKoernerOptions<T> {
    fn default() -> Self {
        Self { max_iter: 50, tol: T::from(1e-8).unwrap() }
    }
}

/// Result of the Sanathanan-Koerner iteration. Converts into its `TransferFunction` with `into`.
#[derive(Debug, Clone)]
pub struct SanathananKoernerResult<T> {
    /// Estimate `G(s) = B(s) / A(s)` of the last step (`A` monic).
    pub transfer_function: TransferFunction<T, Continuous>,
    /// Number of Levy steps run (at most `max_iter`).
    pub iterations: usize,
    /// Whether the relative change of the denominator fell below `tol` (never with `max_iter = 1`).
    pub converged: bool,
    /// Relative change of the denominator at each step from the second,
    /// `max_i |A_k(jω_i) - A_{k-1}(jω_i)| / |A_{k-1}(jω_i)|`.
    pub changes: Vec<T>,
}

/// The estimate of the last step, without the iteration record.
impl<T> From<SanathananKoernerResult<T>> for TransferFunction<T, Continuous> {
    fn from(val: SanathananKoernerResult<T>) -> Self {
        val.transfer_function
    }
}

/// Levy's method (linear least squares on `B(jω) - G(jω) A(jω)` for `G(s) = B(s) / A(s)`):
/// `G(s)` with denominator degree `denom_order` and numerator degree `numer_order`.
/// `omega` of the samples is in rad/s.
///
/// Argument order: `denom_order` (`n`) before `numer_order` (`m`), as in the notation `(n, m)`;
/// e.g. `b_0 / (s^2 + a_1 s + a_2)` is `(2, 0)`.
///
/// Every sample needs a finite `ω >= 0` and a finite `G(jω)`, or `InvalidSample` is returned (also
/// by the Sanathanan-Koerner iteration).
pub fn identify<T: Float + ComplexField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    denom_order: usize,
    numer_order: usize,
) -> Result<TransferFunction<T, Continuous>, LevyIdentificationError> {
    validate(samples)?;
    levy_step(samples, None, numer_order, denom_order, None)
}

/// Levy's method followed by Sanathanan-Koerner reweighting: step `k` minimizes
/// `Σ |G_i A_k(jω_i) - B_k(jω_i)|^2 / |A_{k-1}(jω_i)|^2` (`A_0 = 1`: the first step is Levy's
/// method), which approaches the error `Σ |G_i - B(jω_i) / A(jω_i)|^2` as the iteration converges.
/// The iteration stops when the relative change of the denominator over the samples,
/// `max_i |A_k(jω_i) - A_{k-1}(jω_i)| / |A_{k-1}(jω_i)|` (that of the weights), is below
/// `opts.tol`, or after `opts.max_iter` steps (not an error: see `converged`).
///
/// Argument order: `denom_order` (`n`) before `numer_order` (`m`), as in `identify`.
pub fn sanathanan_koerner_identification<T: Float + ComplexField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    denom_order: usize,
    numer_order: usize,
    opts: &SanathananKoernerOptions<T>,
) -> Result<SanathananKoernerResult<T>, LevyIdentificationError> {
    validate(samples)?;
    iterate(samples, None, denom_order, numer_order, opts)
}

/// `sanathanan_koerner_identification` with one finite, non-negative weight `w_i` per sample: each
/// step minimizes `Σ w_i^2 |G_i A(jω_i) - B(jω_i)|^2 / |A_prev(jω_i)|^2` (`A_prev = 1` in the
/// first step), which approaches the weighted error `Σ w_i^2 |G_i - B(jω_i) / A(jω_i)|^2` as the
/// iteration converges. E.g. `w_i = 1 / |G_i|` for the relative error
/// (`sanathanan_koerner_relative`), or the inverse standard deviation of each bin.
///
/// Argument order: `denom_order` (`n`) before `numer_order` (`m`), as in `identify`.
pub fn sanathanan_koerner_weighted<T: Float + ComplexField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    weights: &[T],
    denom_order: usize,
    numer_order: usize,
    opts: &SanathananKoernerOptions<T>,
) -> Result<SanathananKoernerResult<T>, LevyIdentificationError> {
    validate(samples)?;
    if weights.len() != samples.len() || weights.iter().any(|w| !(Float::is_finite(*w) && *w >= T::zero())) {
        return Err(LevyIdentificationError::InvalidWeights { len: weights.len(), samples: samples.len() });
    }
    iterate(samples, Some(weights), denom_order, numer_order, opts)
}

/// `sanathanan_koerner_identification` by the relative error: `sanathanan_koerner_weighted` with
/// `w_i = 1 / |G_i|`, so that every sample counts alike whatever its gain; as the iteration
/// converges, each step approaches `Σ |G_i - B(jω_i) / A(jω_i)|^2 / |G_i|^2`. A sample with
/// `G_i = 0` (e.g. an unexcited bin of [`welch`](crate::fft::welch)) has no relative error:
/// `ZeroResponse` (the samples are checked first, so a non-finite one is `InvalidSample`).
///
/// Argument order: `denom_order` (`n`) before `numer_order` (`m`), as in `identify`.
pub fn sanathanan_koerner_relative<T: Float + ComplexField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    denom_order: usize,
    numer_order: usize,
    opts: &SanathananKoernerOptions<T>,
) -> Result<SanathananKoernerResult<T>, LevyIdentificationError> {
    validate(samples)?;
    let weights: Vec<T> = samples.iter().map(|s| T::one() / Float::hypot(s.value.re, s.value.im)).collect();
    if let Some(index) = weights.iter().position(|w| !(Float::is_finite(*w) && *w > T::zero())) {
        return Err(LevyIdentificationError::ZeroResponse { index });
    }
    iterate(samples, Some(&weights), denom_order, numer_order, opts)
}

/// Errors of samples that cannot be fitted: none, a non-finite or negative `ω`, or a non-finite
/// response.
fn validate<T: Float>(samples: &[FrequencyResponse<T>]) -> Result<(), LevyIdentificationError> {
    if samples.is_empty() {
        return Err(LevyIdentificationError::EmptyData);
    }
    let valid = |s: &FrequencyResponse<T>| s.omega.is_finite() && s.omega >= T::zero() && s.value.re.is_finite() && s.value.im.is_finite();
    match samples.iter().position(|s| !valid(s)) {
        Some(index) => Err(LevyIdentificationError::InvalidSample { index }),
        None => Ok(()),
    }
}

/// Levy steps, each weighted by the denominator of the previous one, until the relative change of
/// the denominator is below `opts.tol` or `opts.max_iter` steps.
fn iterate<T: Float + ComplexField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    weights: Option<&[T]>,
    denom_order: usize,
    numer_order: usize,
    opts: &SanathananKoernerOptions<T>,
) -> Result<SanathananKoernerResult<T>, LevyIdentificationError> {
    if opts.max_iter == 0 {
        return Err(LevyIdentificationError::IterationError);
    }
    let mut tf = levy_step(samples, weights, numer_order, denom_order, None)?;
    let (mut iterations, mut converged, mut changes) = (1, false, Vec::new());
    while iterations < opts.max_iter {
        let next = levy_step(samples, weights, numer_order, denom_order, Some(tf.clone()))?;
        iterations += 1;
        let change = denominator_change(samples, &tf.denominator.0, &next.denominator.0);
        changes.push(change);
        tf = next;
        if change < opts.tol {
            converged = true;
            break;
        }
    }
    Ok(SanathananKoernerResult { transfer_function: tf, iterations, converged, changes })
}

/// `max_i |A(jω_i) - A_prev(jω_i)| / |A_prev(jω_i)|` (infinite where `A_prev(jω_i) = 0`).
fn denominator_change<T: Float + ComplexField>(samples: &[FrequencyResponse<T>], previous: &[T], current: &[T]) -> T {
    let at = |coeffs: &[T], omega: T| {
        let s = Complex::new(T::zero(), omega);
        coeffs.iter().fold(Complex::new(T::zero(), T::zero()), |acc, &c| acc * s + Complex::new(c, T::zero()))
    };
    samples.iter().fold(T::zero(), |worst, sample| {
        let (old, new) = (at(previous, sample.omega), at(current, sample.omega));
        let d = new - old;
        let change = Float::hypot(d.re, d.im) / Float::hypot(old.re, old.im);
        if Float::is_nan(change) { T::infinity() } else { Float::max(worst, change) }
    })
}

/// One Levy step: rows of sample `i` weighted by `w_i / |A_prev(jω_i)|` (`w_i = 1` without
/// `weights`, `A_prev = 1` without `prev_result`).
fn levy_step<T: Float + ComplexField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    weights: Option<&[T]>,
    numer_order: usize,
    denom_order: usize,
    prev_result: Option<TransferFunction<T, Continuous>>,
) -> Result<TransferFunction<T, Continuous>, LevyIdentificationError> {
    let num_data = samples.len();
    if num_data == 0 {
        return Err(LevyIdentificationError::EmptyData);
    }

    if prev_result.as_ref().is_some_and(|prev| prev.denominator.len() != denom_order + 1) {
        return Err(LevyIdentificationError::OrderMismatch);
    }

    let (num_a, num_b) = (denom_order, numer_order + 1);
    let k = num_a + num_b;

    let mut r = DMatrix::<T>::zeros(k, k);
    let mut rhs = DVector::<T>::zeros(k);

    let s0 = Complex::new(T::one(), T::zero());
    let powers_size = std::cmp::max(denom_order + 1, numer_order);

    for (index, sample) in samples.iter().enumerate() {
        let w = sample.omega;
        let gr = sample.value.re;
        let gi = sample.value.im;

        let s = Complex::<T>::new(T::zero(), w);
        let mut powers = Vec::with_capacity(powers_size);
        let mut current = s0;
        powers.push(current);
        for _ in 1..powers_size {
            current *= s;
            powers.push(current);
        }
        let sn = powers[denom_order];

        let w_p = match &prev_result {
            Some(ret) => {
                let denom_coeffs = &ret.denominator.0;
                let mut denom = Complex::zero();
                for i in 0..(denom_order + 1) {
                    denom += Complex::from(denom_coeffs[denom_order - i]) * powers[i];
                }
                T::one() / denom.norm()
            },
            None => T::one()
        } * weights.map_or(T::one(), |w| w[index]);

        let phi = {
            let mut ret = DMatrix::<T>::zeros(2, k);

            for j in 0..num_a {
                let p = powers[num_a - 1 - j];
                ret[(0, j)] = gr * p.re - gi * p.im;
                ret[(1, j)] = gr * p.im + gi * p.re;
            }

            for j in 0..num_b {
                let p = powers[num_b - 1 - j];
                ret[(0, j + num_a)] = -p.re;
                ret[(1, j + num_a)] = -p.im;
            }

            ret * w_p
        };

        let y = {
            let mut ret = DVector::<T>::zeros(2);
            ret[0] = gr * sn.re - gi * sn.im;
            ret[1] = gr * sn.im + gi * sn.re;
            ret * w_p
        };

        r += &phi.transpose() * &phi;
        rhs += &phi.transpose() * &y;

    }

    let theta = match r.qr().solve(&(-rhs)) {
        Some(x) => x.data.as_slice().to_owned(),
        None => return Err(LevyIdentificationError::SingularMatrix),
    };

    let a_coeffs_desc = &theta[0..num_a];
    let b_coeffs_desc = &theta[num_a..];

    let numer_coeffs = b_coeffs_desc.to_vec();
    let mut denom_coeffs = vec![T::one()];
    denom_coeffs.extend(a_coeffs_desc);

    Ok(TransferFunction::from_polynomials(Polynomial(numer_coeffs), Polynomial(denom_coeffs)))
}
