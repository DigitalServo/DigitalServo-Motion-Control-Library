//! Matched z-transform (pole-zero matching) between continuous-time `G(s)` and discrete-time
//! `G(z)` with sampling period `ts`, in both directions.
//!
//! ```text
//! s -> z :  poles / zeros  p  ->  e^(p ts),   zeros at s = ∞  ->  z = -1 (see `ZerosAtInfinity`)
//! z -> s :  poles / zeros  q  ->  ln(q) / ts, zeros at z = -1 and z = ∞  ->  s = ∞
//! ```
//!
//! The gain is matched at low frequency. With `k` = (poles at the origin) - (zeros at the origin)
//! (integrators, `s = 0` <-> `z = 1`):
//!
//! ```text
//! lim_{s->0} s^k G(s) = lim_{z->1} ((z - 1) / ts)^k G(z)
//! ```
//!
//! which is the DC gain for `k = 0` and the gain of the integrators' asymptote otherwise.
//!
//! In `z -> s`, the factors `(z - 1)` and `(z + 1)` are divided out of the coefficients exactly
//! (synthetic division while the remainder vanishes) rather than detected among numerically found
//! roots, since with fast sampling all poles crowd around `z = 1`. The remaining roots are mapped
//! with the principal logarithm (`|Im s| < π / ts`). Roots at `z = 0` and on the negative real axis
//! have no rational continuous-time counterpart: such zeros are dropped (taken as zeros at `s = ∞`,
//! the low-frequency gain is still matched), such poles are an error. Roots at `z = 0` are time
//! shifts, `z^-d <-> e^(-d ts s)`, and are kept as a delay by `to_continuous_with_delay`.

use std::borrow::Borrow;
use std::ops::AddAssign;

use num_complex::Complex;
use num_traits::{Float, FloatConst};
use thiserror::Error;

use crate::system::roots_with_multiplicity;
use crate::{vieta_formula, Continuous, Discrete, Polynomial, TransferFunction};

/// Where the zeros of `G(s)` at `s = ∞` (one per relative degree) go in `s -> z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZerosAtInfinity {
    /// All of them to `z = -1` (the Nyquist frequency): `G(z)` is biproper.
    MinusOne,
    /// All but one to `z = -1`, one kept at `z = ∞`: `G(z)` is strictly proper (one-sample delay,
    /// leaving a sample period for computation).
    KeepOneDelay,
}

/// Errors of the matched z-transform (`to_discrete`) and of its inverse.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MatchedZError {
    #[error("The transfer function is zero")]
    ZeroSystem,

    #[error("Improper continuous-time transfer function (more zeros than poles) cannot be matched")]
    Improper,

    #[error("Pole at z = 0 (pure delay) has no rational continuous-time counterpart; see `to_continuous_with_delay`")]
    PoleAtOrigin,

    #[error("Pole at z = {re} on the negative real axis has no real continuous-time counterpart")]
    NegativeRealPole { re: f64 },

    #[error("Delay {delay} is not a whole number of sampling periods")]
    FractionalDelay { delay: f64 },
}

/// `G(s) = e^(-delay s) tf(s)`: a rational transfer function with a time shift (dead time for
/// `delay > 0`, time advance for `delay < 0`).
#[derive(Clone, Debug)]
pub struct ContinuousWithDelay<T> {
    pub tf: TransferFunction<T, Continuous>,
    pub delay: T,
}

impl<T: Float> ContinuousWithDelay<T> {
    /// `G(jω) = e^(-jω delay) tf(jω)`.
    pub fn frequency_response(&self, omega: T) -> Complex<T> {
        let s = Complex::new(T::zero(), omega);
        let eval = |p: &Polynomial<T>| p.iter().fold(Complex::new(T::zero(), T::zero()), |acc, &c| acc * s + c);
        (s * -self.delay).exp() * eval(&self.tf.numerator) / eval(&self.tf.denominator)
    }
}

impl<T: Float + std::fmt::Display> std::fmt::Display for ContinuousWithDelay<T> {
    /// e.g. `exp(-0.003 s) * (1000 / (s^2 + 20 * s + 1000))` (precision is passed on).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (delay, tf) = match f.precision() {
            Some(p) => (format!("{:.*}", p, -self.delay), format!("{:.*}", p, self.tf)),
            None => ((-self.delay).to_string(), self.tf.to_string()),
        };
        write!(f, "exp({} s) * ({})", delay, tf)
    }
}

/// Roots within this distance (relative to `max(|root|, 1)`) are one repeated root.
const CLUSTER_TOLERANCE: f64 = 1e-6;
/// Imaginary / real parts within this (relative) size are round-off and set to zero.
const SNAP_TOLERANCE: f64 = 1e-12;
/// `(z ∓ 1)` is a factor while the remainder is within this size relative to `Σ |coefficients|`.
const FACTOR_TOLERANCE: f64 = 1e-10;

/// Continuous-time `G(s)` to discrete-time `G(z)` by pole-zero matching.
pub fn to_discrete<T, S>(tf: S, ts: T, zeros_at_infinity: ZerosAtInfinity) -> Result<TransferFunction<T, Discrete>, MatchedZError>
where
    T: Float + FloatConst + AddAssign,
    S: Borrow<TransferFunction<T, Continuous>>,
{
    let tf = tf.borrow();
    // G(s) = s^(a - b) N(s) / D(s) with N(0), D(0) != 0
    let (numer, a) = split_origin(&tf.numerator).ok_or(MatchedZError::ZeroSystem)?;
    let (denom, b) = split_origin(&tf.denominator).ok_or(MatchedZError::ZeroSystem)?;

    let degree_numer = numer.len() - 1 + a;
    let degree_denom = denom.len() - 1 + b;
    if degree_numer > degree_denom {
        return Err(MatchedZError::Improper);
    }
    let relative_degree = degree_denom - degree_numer;
    let to_minus_one = match zeros_at_infinity {
        ZerosAtInfinity::MinusOne => relative_degree,
        ZerosAtInfinity::KeepOneDelay => relative_degree.saturating_sub(1),
    };

    // Roots away from z = 1 (the integrators are added afterwards).
    let map = |p: &Polynomial<T>| -> Vec<Complex<T>> {
        roots(p).into_iter().flat_map(|(r, m)| std::iter::repeat_n((r * ts).exp(), m)).collect()
    };
    let mut zeros = map(&numer);
    zeros.extend(std::iter::repeat_n(Complex::new(-T::one(), T::zero()), to_minus_one));
    let mut poles = map(&denom);

    // lim s^k G(s) = N(0) / D(0) = lim ((z-1)/ts)^k G(z) = K Π(1 - zero) / Π(1 - pole) / ts^k
    let k = b as i32 - a as i32;
    let asymptote = *numer.last().unwrap() / *denom.last().unwrap();
    let gain = asymptote * ts.powi(k) * product(&poles, T::one()) / product(&zeros, T::one());

    let one = Complex::new(T::one(), T::zero());
    zeros.extend(std::iter::repeat_n(one, a));
    poles.extend(std::iter::repeat_n(one, b));
    Ok(TransferFunction::from_polynomials(real_polynomial(&zeros, gain), real_polynomial(&poles, T::one())))
}

/// Options of `to_continuous_with` for models whose zeros at `z = -1` are not exact,
/// e.g. identified from data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToContinuousOptions<T> {
    /// Zeros within this distance of `z = -1` are zeros at `s = ∞`. Models identified from data
    /// have their zeros at `z = -1` slightly perturbed; a repeated one even splits by about the
    /// square root of the coefficient error (e.g. 1e-8 -> 1e-4 .. 1e-2).
    pub nyquist_tolerance: T,
}

impl<T: Float> Default for ToContinuousOptions<T> {
    /// Exact `(z + 1)` factors only (same as `to_continuous`).
    fn default() -> Self {
        Self { nyquist_tolerance: T::zero() }
    }
}

/// Discrete-time `G(z)` to continuous-time `G(s)` by pole-zero matching (inverse of `to_discrete`).
/// Only exact `(z + 1)` factors of the numerator are taken as zeros at `s = ∞`; for identified
/// models, use `to_continuous_with`.
///
/// Zeros at `z = 0` and on the negative real axis have no rational continuous-time counterpart and
/// are dropped. They appear e.g. as sampling zeros of a zero-order hold or when the model order
/// does not match the plant (a zero far out on the negative real axis is nearly a constant gain
/// at low frequency). The low-frequency gain is still matched, but the phase they contribute
/// (about `ωT` for a zero at `z = 0`, a time advance) is lost; `to_continuous_with_delay` keeps the
/// roots at `z = 0` as a time shift instead.
pub fn to_continuous<T, S>(tf: S, ts: T) -> Result<TransferFunction<T, Continuous>, MatchedZError>
where
    T: Float + FloatConst + AddAssign,
    S: Borrow<TransferFunction<T, Discrete>>,
{
    to_continuous_with(tf, ts, &ToContinuousOptions::default())
}

/// Same as `to_continuous`, with the handling of zeros without an exact continuous-time
/// counterpart given by `options`.
pub fn to_continuous_with<T, S>(tf: S, ts: T, options: &ToContinuousOptions<T>) -> Result<TransferFunction<T, Continuous>, MatchedZError>
where
    T: Float + FloatConst + AddAssign,
    S: Borrow<TransferFunction<T, Discrete>>,
{
    let tf = tf.borrow();
    // Roots at z = 0 (exact: trailing zero coefficients): zeros are dropped (time advance lost),
    // poles are an error.
    let (numer, _) = split_origin(&tf.numerator).ok_or(MatchedZError::ZeroSystem)?;
    let (denom, poles_at_origin) = split_origin(&tf.denominator).ok_or(MatchedZError::ZeroSystem)?;
    if poles_at_origin > 0 {
        return Err(MatchedZError::PoleAtOrigin);
    }
    rational_part(&numer, &denom, ts, options)
}

/// Same as `to_continuous_with`, but roots at `z = 0` are kept as a time shift:
/// `G(z) = z^-d G'(z)  ->  G(s) = e^(-d ts s) G'(s)` (`d` = poles - zeros at `z = 0`; `d > 0` is a
/// dead time, e.g. an input delay of the identified plant).
pub fn to_continuous_with_delay<T, S>(tf: S, ts: T, options: &ToContinuousOptions<T>) -> Result<ContinuousWithDelay<T>, MatchedZError>
where
    T: Float + FloatConst + AddAssign,
    S: Borrow<TransferFunction<T, Discrete>>,
{
    let tf = tf.borrow();
    let (numer, zeros_at_origin) = split_origin(&tf.numerator).ok_or(MatchedZError::ZeroSystem)?;
    let (denom, poles_at_origin) = split_origin(&tf.denominator).ok_or(MatchedZError::ZeroSystem)?;
    let tf = rational_part(&numer, &denom, ts, options)?;
    let shift = poles_at_origin as i32 - zeros_at_origin as i32;
    Ok(ContinuousWithDelay { tf, delay: ts * T::from(shift).unwrap() })
}

/// `e^(-delay s) G(s)` to `z^-d G(z)` with `d = delay / ts`, which must be a whole number.
pub fn to_discrete_with_delay<T>(
    g: &ContinuousWithDelay<T>,
    ts: T,
    zeros_at_infinity: ZerosAtInfinity,
) -> Result<TransferFunction<T, Discrete>, MatchedZError>
where
    T: Float + FloatConst + AddAssign,
{
    let samples = g.delay / ts;
    let d = samples.round();
    if (samples - d).abs() > T::from(1e-6).unwrap() {
        return Err(MatchedZError::FractionalDelay { delay: g.delay.to_f64().unwrap_or(f64::NAN) });
    }
    let d = d.to_i64().unwrap();
    let gz = to_discrete(&g.tf, ts, zeros_at_infinity)?;
    // z^-d: d > 0 multiplies the denominator by z^d, d < 0 the numerator by z^-d
    let times_z_power = |p: &Polynomial<T>, n: i64| {
        let mut c = p.0.clone();
        c.extend(std::iter::repeat_n(T::zero(), n.max(0) as usize));
        Polynomial(c)
    };
    Ok(TransferFunction::from_polynomials(times_z_power(&gz.numerator, -d), times_z_power(&gz.denominator, d)))
}

/// `G(z) = N(z) / D(z)` with `N(0), D(0) != 0` to the rational `G(s)`.
fn rational_part<T>(
    numer: &Polynomial<T>,
    denom: &Polynomial<T>,
    ts: T,
    options: &ToContinuousOptions<T>,
) -> Result<TransferFunction<T, Continuous>, MatchedZError>
where
    T: Float + FloatConst + AddAssign,
{
    // G(z) = (z - 1)^(a - b) N(z) / D(z) with N(1), D(1) != 0
    let (numer, a) = divide_out(numer, T::one());
    let (denom, b) = divide_out(denom, T::one());

    // lim ((z-1)/ts)^k G(z) = N(1) / D(1) / ts^k  (k = b - a)
    let k = b as i32 - a as i32;
    let asymptote = horner(&numer, T::one()) / horner(&denom, T::one()) / ts.powi(k);

    // Zeros at z = -1 go back to s = ∞; the other roots are mapped by ln(q) / ts.
    let (numer, _) = divide_out(&numer, -T::one());
    let minus_one = Complex::new(-T::one(), T::zero());
    let map = |p: &Polynomial<T>, is_numerator: bool| -> Result<Vec<Complex<T>>, MatchedZError> {
        let mut s = Vec::new();
        for (q, m) in roots(p) {
            if is_numerator && (q - minus_one).norm() <= options.nyquist_tolerance {
                continue;
            }
            // No rational continuous-time counterpart: zeros are dropped, poles are an error.
            if q.re.is_zero() && q.im.is_zero() {
                if is_numerator {
                    continue;
                }
                return Err(MatchedZError::PoleAtOrigin);
            }
            if q.im.is_zero() && q.re < T::zero() {
                if is_numerator {
                    continue;
                }
                return Err(MatchedZError::NegativeRealPole { re: q.re.to_f64().unwrap_or(f64::NAN) });
            }
            s.extend(std::iter::repeat_n(q.ln() / ts, m));
        }
        Ok(s)
    };
    let mut zeros = map(&numer, true)?;
    let mut poles = map(&denom, false)?;

    // G(s) = K s^(a - b) Π(s - zero) / Π(s - pole), lim s^k G(s) = K Π(-zero) / Π(-pole) = asymptote
    let gain = asymptote * product(&poles, T::zero()) / product(&zeros, T::zero());

    let origin = Complex::new(T::zero(), T::zero());
    zeros.extend(std::iter::repeat_n(origin, a));
    poles.extend(std::iter::repeat_n(origin, b));
    Ok(TransferFunction::from_polynomials(real_polynomial(&zeros, gain), real_polynomial(&poles, T::one())))
}

/// Leading zeros removed; `None` for the zero polynomial.
fn trim<T: Float>(p: &Polynomial<T>) -> Option<Polynomial<T>> {
    let coeffs: Vec<T> = p.iter().copied().skip_while(|c| c.is_zero()).collect();
    if coeffs.is_empty() { None } else { Some(Polynomial(coeffs)) }
}

/// `p(s) = s^k q(s)` with `q(0) != 0`: returns `(q, k)` (exact: trailing zero coefficients).
fn split_origin<T: Float>(p: &Polynomial<T>) -> Option<(Polynomial<T>, usize)> {
    let mut coeffs = trim(p)?.0;
    let mut k = 0;
    while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
        coeffs.pop();
        k += 1;
    }
    Some((Polynomial(coeffs), k))
}

/// `p(z) = (z - c)^k q(z)`: divide `(z - c)` out while the remainder `p(c)` vanishes
/// (relative to `Σ |coefficients|`). Returns `(q, k)`.
fn divide_out<T: Float>(p: &Polynomial<T>, c: T) -> (Polynomial<T>, usize) {
    let mut coeffs = p.0.clone();
    let mut k = 0;
    let tol = T::from(FACTOR_TOLERANCE).unwrap();
    while coeffs.len() > 1 {
        let scale = coeffs.iter().fold(T::zero(), |acc, &x| acc + x.abs());
        let mut quotient = Vec::with_capacity(coeffs.len());
        let mut acc = T::zero();
        for &x in &coeffs {
            acc = acc * c + x;
            quotient.push(acc);
        }
        let remainder = quotient.pop().unwrap();
        if remainder.abs() > tol * scale {
            break;
        }
        coeffs = quotient;
        k += 1;
    }
    (Polynomial(coeffs), k)
}

fn horner<T: Float>(p: &Polynomial<T>, x: T) -> T {
    p.iter().fold(T::zero(), |acc, &c| acc * x + c)
}

fn roots<T: Float>(p: &Polynomial<T>) -> Vec<(Complex<T>, usize)> {
    if p.len() <= 1 {
        return Vec::new();
    }
    roots_with_multiplicity(p, T::from(CLUSTER_TOLERANCE).unwrap(), T::from(SNAP_TOLERANCE).unwrap())
}

/// `Π (x - r)` (real part; the roots come in conjugate pairs).
fn product<T: Float>(roots: &[Complex<T>], x: T) -> T {
    let x = Complex::new(x, T::zero());
    roots.iter().fold(Complex::new(T::one(), T::zero()), |acc, &r| acc * (x - r)).re
}

/// `gain Π (x - r)` as a real descending polynomial.
fn real_polynomial<T: Float>(roots: &[Complex<T>], gain: T) -> Polynomial<T> {
    Polynomial(vieta_formula(roots).0.iter().map(|c| c.re * gain).collect())
}
