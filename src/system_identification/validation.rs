//! Validation of an identified model on input / output data (e.g. a separate experiment).

use std::ops::AddAssign;

use nalgebra::{ComplexField, RealField};
use num_complex::Complex;
use num_traits::Float;
use thiserror::Error;

use crate::{Continuous, Discrete, Polynomial, TransferFunction};
use crate::system_identification::iv::srivc::{InterSample, Prefilter};

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
}

/// Output-error validation: the model is simulated from rest with the measured input `u`, and
/// compared with the measured output `y` by the residual `ε[k] = y[k] - ŷ[k]`.
///
/// The residual is evaluated from sample `start` on (0 unless set by `evaluated_from`): to validate
/// on the second half of one experiment whose first half was used for the identification, simulate
/// the whole record (so that the state at the split is reproduced) and evaluate the second half.
#[derive(Clone, Debug)]
pub struct Validation<T> {
    /// Measured input `u[k]` over the whole record.
    pub input: Vec<T>,
    /// Simulated output `ŷ[k]` over the whole record.
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

    /// Bayesian information criterion `N ln V + p ln N` with `p = parameters`, the number of
    /// estimated parameters (e.g. `n + m + 1` for `B(s) / A(s)` of degrees `m`, `n`; add 1 if the
    /// input delay was estimated too). The smaller the better; with white Gaussian residuals, a
    /// difference of a few `ln N` is significant.
    pub fn bic(&self, parameters: usize) -> T {
        let n = T::from(self.samples()).unwrap();
        n * self.mse().ln() + T::from(parameters).unwrap() * n.ln()
    }

    /// Test of the residual at the excited lines of a periodic input (e.g. a multisine) of `period`
    /// samples, over the whole periods from `start` on (set `start` after the transient of the
    /// plant from rest, e.g. one period, if the data are also used for the noise level).
    ///
    /// With `E_p(f)` the DFT of the residual over period `p` at line `f` (harmonic of the period),
    /// the model error `(G - Ĝ) U` is the same in every period while the noise varies, so
    ///
    /// ```text
    /// Ē(f) = Σ_p E_p(f) / P,   σ̂^2(f) = Σ_p |E_p(f) - Ē(f)|^2 / (P - 1),   F(f) = P |Ē(f)|^2 / σ̂^2(f)
    /// ```
    ///
    /// `F(f)` compares the model error at the line with the noise level there, estimated from the
    /// data themselves (any noise color). For a right model and Gaussian noise it follows the
    /// F distribution `F(2, 2(P-1))`, whose `confidence` quantile is
    /// `bound = (P-1) ((1 - confidence)^(-1/(P-1)) - 1)`. Every line weighs alike, whatever the
    /// gain of the plant there (unlike the time-domain `mse`).
    pub fn line_test(&self, period: usize, lines: &[usize], confidence: T) -> Result<LineTest<T>, ValidationError> {
        let start = self.start.min(self.residual.len());
        let periods = (self.residual.len() - start) / period.max(1);
        if periods < 2 {
            return Err(ValidationError::TooFewPeriods { periods });
        }
        if let Some(&line) = lines.iter().find(|&&l| l == 0 || 2 * l > period) {
            return Err(ValidationError::InvalidLine { line, half: period / 2 });
        }

        let p = T::from(periods).unwrap();
        // DFT of x over period `index` at `line`
        let dft = |x: &[T], index: usize, line: usize| {
            let offset = start + index * period;
            (0..period).fold(Complex::new(T::zero(), T::zero()), |acc, i| {
                let angle = -T::from(2.0 * std::f64::consts::PI * ((line * i) % period) as f64 / period as f64).unwrap();
                acc + Complex::new(angle.cos(), angle.sin()) * x[offset + i]
            })
        };
        let measured: Vec<T> = self.simulated.iter().zip(&self.residual).map(|(&a, &b)| a + b).collect();

        let mut test = LineTest {
            lines: lines.to_vec(),
            periods,
            residual: Vec::with_capacity(lines.len()),
            output: Vec::with_capacity(lines.len()),
            noise_variance: Vec::with_capacity(lines.len()),
            statistic: Vec::with_capacity(lines.len()),
            bound: (p - T::one()) * ((T::one() - confidence).powf(-T::one() / (p - T::one())) - T::one()),
        };
        for &line in lines {
            let e: Vec<Complex<T>> = (0..periods).map(|index| dft(&self.residual, index, line)).collect();
            let mean = e.iter().fold(Complex::new(T::zero(), T::zero()), |acc, &v| acc + v) / p;
            let variance = e.iter().fold(T::zero(), |acc, &v| acc + (v - mean).norm_sqr()) / (p - T::one());
            let output = (0..periods).fold(Complex::new(T::zero(), T::zero()), |acc, index| acc + dft(&measured, index, line)) / p;
            test.statistic.push(p * mean.norm_sqr() / variance);
            test.residual.push(mean);
            test.output.push(output);
            test.noise_variance.push(variance);
        }
        Ok(test)
    }

    /// Cross-correlation test of the residual and the input over the lags `τ = -max_lag ..= max_lag`.
    ///
    /// If the model is right, the residual is the measurement noise alone and is independent of the
    /// input. The normalized cross-correlation (means removed, over the evaluated samples)
    ///
    /// ```text
    /// r(τ) = R_εu(τ) / sqrt(R_ε(0) R_u(0)),   R_εu(τ) = Σ_k ε[k] u[k-τ] / N
    /// ```
    ///
    /// is then asymptotically normal with zero mean and variance `P / (N R_ε(0) R_u(0))`,
    /// `P = Σ_k R_ε(k) R_u(k)`, which holds for a colored residual and a colored input alike
    /// (`P = R_ε(0) R_u(0)` if either is white). `P` is estimated over `|k| <= max_lag` with a
    /// Bartlett window (so that it is never negative). The bound is `z` standard deviations
    /// (`z = 1.96` for 95 %, `2.58` for 99 % per lag).
    ///
    /// `r(τ)` outside the bound at `τ > 0` (past inputs) indicates unmodeled dynamics or a wrong
    /// delay; at `τ < 0` (future inputs), feedback from the output to the input in the data.
    pub fn cross_correlation(&self, max_lag: usize, z: T) -> CrossCorrelation<T> {
        let (start, end) = (self.start.min(self.residual.len()), self.residual.len());
        let n = T::from(end - start).unwrap();
        let mean = |x: &[T]| x.iter().fold(T::zero(), |acc, &v| acc + v) / n;
        let (e_mean, u_mean) = (mean(&self.residual[start..]), mean(&self.input[start..]));
        let e: Vec<T> = self.residual.iter().map(|&v| v - e_mean).collect();
        let u: Vec<T> = self.input.iter().map(|&v| v - u_mean).collect();

        // Σ_{k in evaluated range, 0 <= k - τ < len} a[k] b[k - τ] / N
        let correlate = |a: &[T], b: &[T], tau: isize, from: usize| {
            let first = (start as isize).max(tau + from as isize) as usize;
            let last = (end as isize).min(end as isize + tau).max(first as isize) as usize;
            (first..last).fold(T::zero(), |acc, k| acc + a[k] * b[(k as isize - tau) as usize]) / n
        };
        let max_lag = max_lag as isize;
        let lags: Vec<isize> = (-max_lag..=max_lag).collect();
        // The input before `start` is known (past of the evaluated samples); the residual is not used there
        let r_eu: Vec<T> = lags.iter().map(|&tau| correlate(&e, &u, tau, 0)).collect();
        let r_e: Vec<T> = (0..=max_lag).map(|k| correlate(&e, &e, k, start)).collect();
        let r_u: Vec<T> = (0..=max_lag).map(|k| correlate(&u, &u, k, start)).collect();

        let scale = (r_e[0] * r_u[0]).sqrt();
        let p = (1..=max_lag as usize).fold(r_e[0] * r_u[0], |acc, k| {
            let w = T::one() - T::from(k).unwrap() / T::from(max_lag + 1).unwrap();
            acc + (T::one() + T::one()) * w * r_e[k] * r_u[k]
        });
        CrossCorrelation {
            lags,
            correlation: r_eu.iter().map(|&r| r / scale).collect(),
            bound: z * (p.max(T::zero()) / n).sqrt() / scale,
        }
    }
}

/// Result of `Validation::line_test`.
#[derive(Clone, Debug)]
pub struct LineTest<T> {
    /// Excited lines (harmonic numbers of the period).
    pub lines: Vec<usize>,
    /// Number of whole periods `P` evaluated.
    pub periods: usize,
    /// Residual spectrum averaged over the periods, `Ē(f) = Σ_p E_p(f) / P`, at each line.
    pub residual: Vec<Complex<T>>,
    /// Measured output spectrum averaged over the periods, `Ȳ(f)`, at each line.
    pub output: Vec<Complex<T>>,
    /// Sample variance of `E_p(f)` over the periods (noise variance) at each line.
    pub noise_variance: Vec<T>,
    /// `F(f) = P |Ē(f)|^2 / σ̂^2(f)`, distributed as `F(2, 2(P-1))` for a right model.
    pub statistic: Vec<T>,
    /// Bound of `F(f)` at the given confidence.
    pub bound: T,
}

impl<T: Float> LineTest<T> {
    /// Lines with `F(f) > bound`.
    pub fn outside(&self) -> Vec<usize> {
        self.lines.iter().zip(&self.statistic).filter(|(_, f)| **f > self.bound).map(|(&l, _)| l).collect()
    }

    /// Fraction of the lines with `F(f) > bound`; about `1 - confidence` for a right model.
    pub fn fraction_outside(&self) -> T {
        T::from(self.outside().len()).unwrap() / T::from(self.lines.len()).unwrap()
    }

    /// Mean of `F(f)` over the lines, to compare with its expectation for a right model
    /// (`expected_statistic`): the model error relative to the noise level, all lines alike.
    pub fn mean_statistic(&self) -> T {
        self.statistic.iter().fold(T::zero(), |acc, &f| acc + f) / T::from(self.lines.len()).unwrap()
    }

    /// Expectation `ν / (ν - 2)` of `F(2, ν)`, `ν = 2(P-1)` (infinite for `P = 2`).
    pub fn expected_statistic(&self) -> T {
        let nu = T::from(2 * (self.periods - 1)).unwrap();
        nu / (nu - T::from(2).unwrap())
    }

    /// RMS over the lines of the relative error `|Ē(f)| / |Ȳ(f)|` (≈ `|Ĝ - G| / |G|` at the line,
    /// plus the noise of the average).
    pub fn rms_relative_error(&self) -> T {
        let sum = self.residual.iter().zip(&self.output).fold(T::zero(), |acc, (e, y)| acc + e.norm_sqr() / y.norm_sqr());
        (sum / T::from(self.lines.len()).unwrap()).sqrt()
    }
}

/// Result of `Validation::cross_correlation`.
#[derive(Clone, Debug)]
pub struct CrossCorrelation<T> {
    /// Lags `τ = -max_lag ..= max_lag` \[samples\].
    pub lags: Vec<isize>,
    /// Normalized cross-correlation `r(τ)` of the residual and the input at each lag.
    pub correlation: Vec<T>,
    /// Confidence bound: `|r(τ)| <= bound` is consistent with a residual independent of the input.
    pub bound: T,
}

impl<T: Float> CrossCorrelation<T> {
    /// Lags with `|r(τ)| > bound`.
    pub fn outside(&self) -> Vec<isize> {
        self.lags.iter().zip(&self.correlation).filter(|(_, r)| r.abs() > self.bound).map(|(&tau, _)| tau).collect()
    }

    /// Fraction of the lags with `|r(τ)| > bound`; about the nominal level (5 % for `z = 1.96`)
    /// for a right model, since every lag is tested separately.
    pub fn fraction_outside(&self) -> T {
        T::from(self.outside().len()).unwrap() / T::from(self.lags.len()).unwrap()
    }

    /// `max |r(τ)| / bound` over the lags with `τ >= 0` (past inputs). Far above 1 (or many lags
    /// above 1) means the residual still depends on the input; slightly above 1 at a few lags is
    /// expected by chance when many lags are tested.
    pub fn max_ratio(&self) -> T {
        self.lags.iter().zip(&self.correlation).filter(|(tau, _)| **tau >= 0).fold(T::zero(), |acc, (_, r)| acc.max(r.abs())) / self.bound
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
