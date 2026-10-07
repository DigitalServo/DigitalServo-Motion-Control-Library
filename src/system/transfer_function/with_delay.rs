//! `TransferFunctionWithDelay`: a continuous-time transfer function with a dead time.

use std::ops::AddAssign;

use nalgebra::{ComplexField, RealField};
use num_complex::Complex;
use num_traits::Float;
use thiserror::Error;

use crate::{Continuous, Polynomial, TransferFunction};
use crate::sampling::whole_samples;
use crate::discretize::InterSample;
use crate::discretize::state_variable_filter::StateVariableFilter;

/// Errors of `TransferFunctionWithDelay::simulate`.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SimulationError {
    #[error("improper model: numerator degree {numerator} > denominator degree {denominator}")]
    Improper { numerator: usize, denominator: usize },
    #[error("zero denominator")]
    ZeroDenominator,
    #[error("delay {delay} is not a whole number of sampling periods")]
    FractionalDelay { delay: f64 },
    #[error("negative delay {delay} (a time advance) cannot be simulated")]
    NegativeDelay { delay: f64 },
    #[error("sampling period {ts} is not positive and finite")]
    InvalidSamplingPeriod { ts: f64 },
}

/// `G(s) = e^(-delay s) tf(s)`: a rational transfer function with a time shift (dead time for
/// `delay > 0`, time advance for `delay < 0`), e.g. a plant with a transport or computation delay,
/// or a model identified with an input delay (`srivc::SrivcResult::model`).
#[derive(Clone, Debug)]
pub struct TransferFunctionWithDelay<T> {
    /// Rational part.
    pub tf: TransferFunction<T, Continuous>,
    /// Dead time \[s\] (negative for a time advance).
    pub delay: T,
}

impl<T> TransferFunctionWithDelay<T> {
    /// `e^(-delay s) tf(s)`.
    pub fn new(tf: TransferFunction<T, Continuous>, delay: T) -> Self {
        Self { tf, delay }
    }
}

impl<T: Float> TransferFunctionWithDelay<T> {
    /// `G(jω) = e^(-jω delay) tf(jω)`.
    pub fn frequency_response(&self, omega: T) -> Complex<T> {
        let s = Complex::new(T::zero(), omega);
        let eval = |p: &Polynomial<T>| p.iter().fold(Complex::new(T::zero(), T::zero()), |acc, &c| acc * s + c);
        (s * -self.delay).exp() * eval(&self.tf.numerator) / eval(&self.tf.denominator)
    }
}

impl<T: Float + AddAssign + ComplexField + RealField> TransferFunctionWithDelay<T> {
    /// Dead time in sampling periods `delay / ts`, if it is a whole number (relative tolerance
    /// `max(1e-9, 4 eps)`).
    pub fn delay_samples(&self, ts: T) -> Result<usize, SimulationError> {
        let delay = self.delay.to_f64().unwrap_or(f64::NAN);
        if !(ts > T::zero() && Float::is_finite(ts)) {
            return Err(SimulationError::InvalidSamplingPeriod { ts: ts.to_f64().unwrap_or(f64::NAN) });
        }
        if self.delay < T::zero() {
            return Err(SimulationError::NegativeDelay { delay });
        }
        // Not a whole number, or not finite
        whole_samples(self.delay, ts).map_err(|_| SimulationError::FractionalDelay { delay })
    }

    /// Sampled response `y[k] = y(k ts)` to the input `u[k]` applied through a zero-order hold,
    /// from rest (the dead time must be a whole number of sampling periods).
    ///
    /// Exact for the held input. `y = B(s) x`, `x = u / A(s)` (`A` monic), with `x` and its
    /// derivatives from the exactly discretized state-variable filter `1 / A(s)` in states scaled by
    /// the root radius of `A(s)`, so that high orders (coefficients spanning many decades) stay
    /// accurate, unlike a controllable canonical realization.
    pub fn simulate(&self, ts: T, u: &[T]) -> Result<Vec<T>, SimulationError> {
        let trim = |p: &Polynomial<T>| p.iter().copied().skip_while(|c| c.is_zero()).collect::<Vec<T>>();
        let (numer, denom) = (trim(&self.tf.numerator), trim(&self.tf.denominator));
        let Some(&lead) = denom.first() else {
            return Err(SimulationError::ZeroDenominator);
        };
        if numer.len() > denom.len() {
            return Err(SimulationError::Improper { numerator: numer.len() - 1, denominator: denom.len() - 1 });
        }
        let nk = self.delay_samples(ts)?;
        let numer: Vec<T> = if numer.is_empty() { vec![T::zero()] } else { numer.iter().map(|&c| c / lead).collect() };
        let denom: Vec<T> = denom.iter().map(|&c| c / lead).collect();
        let (m, n) = (numer.len() - 1, denom.len() - 1);
        let delayed: Vec<T> = (0..u.len()).map(|k| if k >= nk { u[k - nk] } else { T::zero() }).collect();
        if n == 0 {
            return Ok(delayed.iter().map(|&v| numer[0] * v).collect());
        }
        let xf = StateVariableFilter::new(&denom[1..], ts).apply(&delayed, InterSample::ZeroOrderHold);
        Ok((0..u.len()).map(|k| (0..=m).fold(T::zero(), |acc, j| acc + numer[j] * xf[(k, m - j)])).collect())
    }
}

impl<T: Float> From<TransferFunction<T, Continuous>> for TransferFunctionWithDelay<T> {
    /// No dead time.
    fn from(tf: TransferFunction<T, Continuous>) -> Self {
        Self { tf, delay: T::zero() }
    }
}

impl<T: Float> From<&TransferFunction<T, Continuous>> for TransferFunctionWithDelay<T> {
    /// No dead time.
    fn from(tf: &TransferFunction<T, Continuous>) -> Self {
        Self { tf: tf.clone(), delay: T::zero() }
    }
}

impl<T: Clone> From<&TransferFunctionWithDelay<T>> for TransferFunctionWithDelay<T> {
    fn from(g: &TransferFunctionWithDelay<T>) -> Self {
        g.clone()
    }
}

impl<T: Float + std::fmt::Display> std::fmt::Display for TransferFunctionWithDelay<T> {
    /// e.g. `exp(-0.003 s) * (1000 / (s^2 + 20 * s + 1000))` (precision is passed on).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (delay, tf) = match f.precision() {
            Some(p) => (format!("{:.*}", p, -self.delay), format!("{:.*}", p, self.tf)),
            None => ((-self.delay).to_string(), self.tf.to_string()),
        };
        write!(f, "exp({} s) * ({})", delay, tf)
    }
}
