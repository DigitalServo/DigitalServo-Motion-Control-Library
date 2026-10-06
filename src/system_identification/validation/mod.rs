//! Validation of an identified model on input / output data (e.g. a separate experiment).
//!
//! `Validation` holds the residual `ε = y - ŷ` of a model (output error or one-step prediction
//! error); the criteria and tests on it are in the submodules:
//!
//! | Module | Methods | Result |
//! | --- | --- | --- |
//! | `information_criteria` | `bic`, `aic`, `aicc` | |
//! | `whiteness` | `autocorrelation` | `WhitenessTest` |
//! | `cross_correlation` | `cross_correlation` | `CrossCorrelation` |
//! | `line_test` | `line_test` (periodic input) | `LineTest` |
//! | `coherence` | `coherence_test` (any input) | `CoherenceTest` |
//! | `frequency_response` | `frequency_response` (diagnostic) | `FrequencyResponseComparison` |
//!
//! (`spectra`: the segment spectra shared by `coherence` and `frequency_response`.)

use std::ops::AddAssign;

use nalgebra::{ComplexField, RealField, Scalar};
use num_traits::Float;
use thiserror::Error;

use crate::{Continuous, Discrete, Polynomial, TransferFunction};
use crate::system_identification::arx::Arx;
use crate::system_identification::iv::srivc::{InterSample, Prefilter};

mod coherence;
mod information_criteria;
mod cross_correlation;
mod frequency_response;
mod line_test;
mod spectra;
mod whiteness;

pub use coherence::CoherenceTest;
pub use cross_correlation::CrossCorrelation;
pub use frequency_response::FrequencyResponseComparison;
pub use line_test::LineTest;
pub use whiteness::WhitenessTest;

/// Errors of `Validation`.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ValidationError {
    #[error("input and output lengths differ: {input} vs {output}")]
    LengthMismatch { input: usize, output: usize },
    #[error("improper model: numerator degree {numerator} > denominator degree {denominator}")]
    Improper { numerator: usize, denominator: usize },
    #[error("zero denominator")]
    ZeroDenominator,
    #[error("{periods} whole period(s) evaluated; at least 2 are needed")]
    TooFewPeriods { periods: usize },
    #[error("line {line} is not between 1 and period / 2 = {half}")]
    InvalidLine { line: usize, half: usize },
    #[error("{segments} segment(s) of {segment_len} samples evaluated; at least 2 are needed")]
    TooFewSegments { segments: usize, segment_len: usize },
}

/// Validation of a model on measured data by the residual `ε[k] = y[k] - ŷ[k]`, with `ŷ` either
///
/// - the output of the model simulated from rest with the measured input `u` (output error:
///   `continuous`, `discrete`), which validates the input-output model `G`; or
/// - the one-step prediction from the measured past outputs and inputs (prediction error:
///   `one_step_prediction`), which validates a model together with its noise model (ARX).
///
/// `bic`, `aic`, `autocorrelation` and `cross_correlation` apply to both. `line_test`,
/// `coherence_test` and `frequency_response` look at the input-output relation of `ŷ` and are
/// meant for the output error.
///
/// The residual is evaluated from sample `start` on (0 unless set by `evaluated_from`): to validate
/// on the second half of one experiment whose first half was used for the identification, simulate
/// the whole record (so that the state at the split is reproduced) and evaluate the second half.
#[derive(Clone, Debug)]
pub struct Validation<T> {
    /// Measured input `u[k]` over the whole record.
    pub input: Vec<T>,
    /// Model output `ŷ[k]` over the whole record (simulated, or one-step predicted).
    pub simulated: Vec<T>,
    /// Residual `ε[k] = y[k] - ŷ[k]` over the whole record.
    pub residual: Vec<T>,
    /// First evaluated sample.
    pub start: usize,
}

impl<T: Float> Validation<T> {
    /// Continuous-time model `e^(-nk ts s) G(s)` (`nk = input_delay` \[samples\]) driven by `u`
    /// through a zero-order hold, sampled with period `ts` (exact discretization).
    pub fn continuous(
        model: &TransferFunction<T, Continuous>,
        input_delay: usize,
        ts: T,
        u: &[T],
        y: &[T],
    ) -> Result<Self, ValidationError>
    where
        T: AddAssign + ComplexField + RealField,
    {
        let (numer, denom) = monic(&model.numerator, &model.denominator)?;
        let (m, n) = (numer.len() - 1, denom.len() - 1);
        let u_delayed = delayed(u, input_delay);
        let simulated = if n == 0 {
            u_delayed.iter().map(|&v| numer[0] * v).collect()
        } else {
            // ŷ = B(s) x, x = u / A(s): x and its derivatives from the (balanced) filter 1 / A(s)
            let xf = Prefilter::new(&denom[1..], ts).apply(&u_delayed, InterSample::ZeroOrderHold);
            (0..u.len())
                .map(|k| (0..=m).fold(T::zero(), |acc, j| acc + numer[j] * xf[(k, m - j)]))
                .collect()
        };
        Self::new(u, simulated, y)
    }

    /// One-step prediction error of an ARX model (its parameters `arx.parameter`, orders and input
    /// delay; its history is not used): from rest,
    ///
    /// ```text
    /// ŷ[k|k-1] = Σ_i a_i y[k-i] + Σ_j b_j u[k-nk-j] = φ[k]ᵀ θ
    /// ```
    ///
    /// with the measured past outputs, as in the identification (`Arx::push`, `Arx::predict`).
    /// The ARX model `A(z) y = B(z) u + e` assumes a white equation error `e`, i.e. noise
    /// `e / A(z)` on the output: the model (with this noise model) is right only if the prediction
    /// error `ε = A(z) y - B(z) u` is white (`autocorrelation`) and independent of the input
    /// (`cross_correlation`). With white noise on the output instead (`y = G u + v`), even the true
    /// `A`, `B` leave the colored error `A(z) v`: the ARX noise model does not fit, and the IV
    /// method (validated by the output error, `discrete`) is the one to use.
    pub fn one_step_prediction(arx: &Arx<T>, u: &[T], y: &[T]) -> Result<Self, ValidationError>
    where
        T: AddAssign + Scalar,
    {
        if u.len() != y.len() {
            return Err(ValidationError::LengthMismatch { input: u.len(), output: y.len() });
        }
        let mut model = arx.clone();
        model.clear_history();
        let predicted = (0..u.len())
            .map(|k| {
                model.push(u[k], if k > 0 { y[k - 1] } else { T::zero() });
                model.predict()
            })
            .collect();
        Self::new(u, predicted, y)
    }

    /// Discrete-time model `G(z)` driven by `u`.
    pub fn discrete(model: &TransferFunction<T, Discrete>, u: &[T], y: &[T]) -> Result<Self, ValidationError> {
        let (numer, denom) = monic(&model.numerator, &model.denominator)?;
        // G = z^-r (Σ n_i z^-i) / (1 + Σ d_i z^-i), r = deg D - deg N
        let r = denom.len() - numer.len();
        let mut simulated: Vec<T> = Vec::with_capacity(u.len());
        for k in 0..u.len() {
            let mut yk = T::zero();
            for (i, &c) in numer.iter().enumerate() {
                if k >= r + i {
                    yk = yk + c * u[k - r - i];
                }
            }
            for (i, &c) in denom.iter().enumerate().skip(1) {
                if k >= i {
                    yk = yk - c * simulated[k - i];
                }
            }
            simulated.push(yk);
        }
        Self::new(u, simulated, y)
    }

    fn new(u: &[T], simulated: Vec<T>, y: &[T]) -> Result<Self, ValidationError> {
        if u.len() != y.len() {
            return Err(ValidationError::LengthMismatch { input: u.len(), output: y.len() });
        }
        let residual = y.iter().zip(&simulated).map(|(&a, &b)| a - b).collect();
        Ok(Self { input: u.to_vec(), simulated, residual, start: 0 })
    }

    /// Evaluate the residual from sample `start` on.
    pub fn evaluated_from(mut self, start: usize) -> Self {
        self.start = start;
        self
    }

    /// Evaluated residual `ε[start..]`.
    pub fn evaluated(&self) -> &[T] {
        &self.residual[self.start.min(self.residual.len())..]
    }

    /// Number of evaluated samples `N`.
    pub fn samples(&self) -> usize {
        self.evaluated().len()
    }

    /// Mean squared residual `V = Σ ε^2 / N`.
    pub fn mse(&self) -> T {
        let e = self.evaluated();
        e.iter().fold(T::zero(), |acc, &v| acc + v * v) / T::from(e.len()).unwrap()
    }
}

/// Numerator and denominator without leading zeros, divided by the leading denominator coefficient.
fn monic<T: Float>(numerator: &Polynomial<T>, denominator: &Polynomial<T>) -> Result<(Vec<T>, Vec<T>), ValidationError> {
    let trim = |p: &Polynomial<T>| p.iter().copied().skip_while(|c| c.is_zero()).collect::<Vec<T>>();
    let (numer, denom) = (trim(numerator), trim(denominator));
    let Some(&lead) = denom.first() else {
        return Err(ValidationError::ZeroDenominator);
    };
    if numer.len() > denom.len() {
        return Err(ValidationError::Improper { numerator: numer.len() - 1, denominator: denom.len() - 1 });
    }
    let numer = if numer.is_empty() { vec![T::zero()] } else { numer };
    Ok((numer.iter().map(|&c| c / lead).collect(), denom.iter().map(|&c| c / lead).collect()))
}

/// `u` delayed by `delay` samples (zero before).
fn delayed<T: Float>(u: &[T], delay: usize) -> Vec<T> {
    (0..u.len()).map(|k| if k >= delay { u[k - delay] } else { T::zero() }).collect()
}
