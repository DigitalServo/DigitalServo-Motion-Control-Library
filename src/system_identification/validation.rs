//! Validation of an identified model on input / output data (e.g. a separate experiment).

use std::ops::AddAssign;

use nalgebra::{ComplexField, RealField, Scalar};
use num_complex::Complex;
use num_traits::Float;
use rustfft::{FftNum, FftPlanner};
use thiserror::Error;

use crate::{Continuous, Discrete, Polynomial, TransferFunction};
use crate::system_identification::arx::Arx;
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

    /// Bayesian information criterion `N ln V + p ln N` with `p = parameters`, the number of
    /// estimated parameters (e.g. `n + m + 1` for `B(s) / A(s)` of degrees `m`, `n`; add 1 if the
    /// input delay was estimated too). The smaller the better; with white Gaussian residuals, a
    /// difference of a few `ln N` is significant.
    pub fn bic(&self, parameters: usize) -> T {
        let n = T::from(self.samples()).unwrap();
        n * self.mse().ln() + T::from(parameters).unwrap() * n.ln()
    }

    /// Akaike information criterion `N ln V + 2p` (`p = parameters` as in `bic`). Its penalty does
    /// not grow with `N`, so with long records it tends to prefer more parameters than BIC (it
    /// aims at the best prediction, not at the true structure).
    pub fn aic(&self, parameters: usize) -> T {
        let n = T::from(self.samples()).unwrap();
        n * self.mse().ln() + T::from(2 * parameters).unwrap()
    }

    /// AIC corrected for small samples, `AIC + 2p(p + 1) / (N - p - 1)` (infinite for
    /// `N <= p + 1`); the same as `aic` for `N >> p^2`.
    pub fn aicc(&self, parameters: usize) -> T {
        let (n, p) = (self.samples(), parameters);
        if n <= p + 1 {
            return T::infinity();
        }
        self.aic(p) + T::from(2 * p * (p + 1)).unwrap() / T::from(n - p - 1).unwrap()
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

    /// Whiteness test of the residual over the lags `τ = 1 ..= max_lag`: the normalized
    /// autocorrelation (mean removed, over the evaluated samples)
    ///
    /// ```text
    /// r(τ) = R_ε(τ) / R_ε(0),   R_ε(τ) = Σ_k ε[k] ε[k-τ] / N
    /// ```
    ///
    /// is asymptotically normal with zero mean and variance `1 / N` for a white residual, so each
    /// lag is tested against `±z / sqrt(N)` (`z = 2.58` for 99 % per lag). All lags together are
    /// tested by the Ljung-Box statistic `Q = N (N + 2) Σ_τ r(τ)^2 / (N - τ)`, distributed as
    /// `χ²(max_lag)` for a white residual (`WhitenessTest::ljung_box_bound`).
    ///
    /// What a colored residual means depends on the residual:
    /// - output error (a simulated model, e.g. SRIVC): the residual is the measurement noise itself
    ///   when `G` is right, so a colored residual says that the noise is colored, not that `G` is
    ///   wrong (test `G` by `cross_correlation` / `coherence_test`); it questions the white-noise
    ///   assumptions of the estimator and of `bic` / `aic`, and may call for a noise model;
    /// - one-step prediction error (a model with its noise model, e.g. ARX): the model is right
    ///   only if the prediction error is white.
    pub fn autocorrelation(&self, max_lag: usize, z: T) -> WhitenessTest<T> {
        let start = self.start.min(self.residual.len());
        let e = &self.residual[start..];
        let n = T::from(e.len()).unwrap();
        let mean = e.iter().fold(T::zero(), |acc, &v| acc + v) / n;
        let e: Vec<T> = e.iter().map(|&v| v - mean).collect();
        let r = |tau: usize| (tau..e.len()).fold(T::zero(), |acc, k| acc + e[k] * e[k - tau]) / n;

        let r0 = r(0);
        let lags: Vec<usize> = (1..=max_lag).collect();
        let correlation: Vec<T> = lags.iter().map(|&tau| r(tau) / r0).collect();
        let ljung_box = n * (n + T::from(2).unwrap())
            * lags.iter().zip(&correlation).fold(T::zero(), |acc, (&tau, &c)| acc + c * c / (n - T::from(tau).unwrap()));
        WhitenessTest { lags, correlation, bound: z / n.sqrt(), ljung_box }
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

impl<T: Float + FftNum> Validation<T> {
    /// Coherence test of the residual and the input, for any input (non-periodic: random, chirp,
    /// measured operation data, ...).
    ///
    /// The evaluated samples are split into segments of `segment_len` samples (Hann window, 50 %
    /// overlap, mean removed per segment), and with the DFTs `E_l(f)`, `U_l(f)` of segment `l`
    ///
    /// ```text
    /// γ²(f) = |Σ_l E_l(f) U_l(f)*|^2 / (Σ_l |E_l(f)|^2  Σ_l |U_l(f)|^2)
    /// ```
    ///
    /// at the bins `f = 1 ..= segment_len / 2` (frequency `f / (segment_len ts)`). If the model is
    /// right, the residual is noise independent of the input, and for `L` independent segments
    /// `γ²(f)` follows the Beta distribution `Beta(1, L - 1)` whatever the input (it is the squared
    /// projection of the Gaussian vector `E_l(f)` on the fixed direction `U_l(f)`): the bound at
    /// the given `confidence` is `1 - (1 - confidence)^(1 / (L - 1))`, and the mean `1 / L`.
    /// Overlapping segments are not independent; `L` is the equivalent number of independent
    /// segments for the window and the overlap (Welch: `K / (1 + 2 Σ_m (1 - m/K) c_m^2)` for `K`
    /// segments, `c_m` the normalized overlap of the window with itself shifted by `m` steps).
    ///
    /// `γ²(f)` outside the bound means the residual still depends on the input at that frequency.
    /// Longer segments give a finer frequency resolution but fewer segments, so a higher bound
    /// (less power); at the frequencies where the input has no power the test cannot detect
    /// anything (see `CoherenceTest::excited`).
    pub fn coherence_test(&self, segment_len: usize, confidence: T) -> Result<CoherenceTest<T>, ValidationError> {
        let spectra = self.spectra(segment_len)?;
        let l = spectra.effective_segments;
        let bins: Vec<usize> = (1..=segment_len / 2).collect();
        Ok(CoherenceTest {
            coherence: bins
                .iter()
                .map(|&f| {
                    let denom = spectra.ee[f] * spectra.uu[f];
                    if denom > T::zero() { spectra.eu[f].norm_sqr() / denom } else { T::zero() }
                })
                .collect(),
            input_power: bins.iter().map(|&f| spectra.uu[f]).collect(),
            bins,
            segment_len,
            segments: spectra.segments,
            effective_segments: l,
            bound: T::one() - (T::one() - confidence).powf(T::one() / (l - T::one())),
        })
    }

    /// Frequency response of the model compared with the nonparametric estimate from the data,
    /// for any input, by the same segments as `coherence_test`:
    ///
    /// ```text
    /// measured  G(f) = S_yu(f) / S_uu(f)              (H1 estimate)
    /// model     Ĝ(f) = S_ŷu(f) / S_uu(f) = G(f) - S_εu(f) / S_uu(f)
    /// ```
    ///
    /// Both are estimated with the same windowed segments, so the leakage of the window affects
    /// them alike and their difference is exactly `S_εu / S_uu` (the model response includes the
    /// input delay, from the simulation). Both are therefore windowed estimates of the response of
    /// the sampled system (with the hold), not `Ĝ(jω)`: their bias, as for any H1 estimate, falls
    /// as the segments grow against the impulse response of the plant. The standard deviation of the measured response is
    /// `σ(f) = sqrt((1 - γ²_yu) S_yy / (L S_uu))` (`γ²_yu` the coherence of the output and the
    /// input, `L` the equivalent number of independent segments): `G ± 2σ` is a ~95 % band, and
    /// `|Ĝ - G|^2 / σ^2` is about `χ²(2) / 2` (mean 1) for a right model. Where the leakage is
    /// strong (sharp resonances against short segments) it also lowers `γ²_yu` and so inflates
    /// `σ`: the normalized error is then conservative (below 1 for a right model).
    ///
    /// This is a diagnostic rather than another test (it carries the same information as
    /// `coherence_test`): it shows the model error as a gain \[dB\] and a phase error against the
    /// uncertainty of the data, e.g. for a Bode plot of both.
    pub fn frequency_response(&self, segment_len: usize) -> Result<FrequencyResponseComparison<T>, ValidationError> {
        let spectra = self.spectra(segment_len)?;
        let l = spectra.effective_segments;
        let zero = Complex::new(T::zero(), T::zero());
        let bins: Vec<usize> = (1..=segment_len / 2).collect();
        let ratio = |a: Complex<T>, f: usize| if spectra.uu[f] > T::zero() { a / spectra.uu[f] } else { zero };
        Ok(FrequencyResponseComparison {
            measured: bins.iter().map(|&f| ratio(spectra.yu[f], f)).collect(),
            model: bins.iter().map(|&f| ratio(spectra.yu[f] - spectra.eu[f], f)).collect(),
            coherence: bins
                .iter()
                .map(|&f| {
                    let denom = spectra.uu[f] * spectra.yy[f];
                    if denom > T::zero() { spectra.yu[f].norm_sqr() / denom } else { T::zero() }
                })
                .collect(),
            stdev: bins
                .iter()
                .map(|&f| {
                    if spectra.uu[f] > T::zero() {
                        let noise = spectra.yy[f] - spectra.yu[f].norm_sqr() / spectra.uu[f]; // (1 - γ²) S_yy
                        (noise.max(T::zero()) / (l * spectra.uu[f])).sqrt()
                    } else {
                        T::infinity()
                    }
                })
                .collect(),
            input_power: bins.iter().map(|&f| spectra.uu[f]).collect(),
            bins,
            segment_len,
            effective_segments: l,
        })
    }

    /// Mean auto / cross spectra over the segments of `segment_len` samples (Hann window, 50 %
    /// overlap, mean removed per segment) of the residual `ε`, the input `u` and the output `y`,
    /// at the bins `0 ..= segment_len / 2`, and the equivalent number of independent segments.
    fn spectra(&self, segment_len: usize) -> Result<Spectra<T>, ValidationError> {
        let start = self.start.min(self.residual.len());
        let (e, u) = (&self.residual[start..], &self.input[start..]);
        let y: Vec<T> = self.simulated[start..].iter().zip(e).map(|(&a, &b)| a + b).collect();
        let step = (segment_len / 2).max(1);
        let segments = if segment_len >= 2 && e.len() >= segment_len { (e.len() - segment_len) / step + 1 } else { 0 };
        if segments < 2 {
            return Err(ValidationError::TooFewSegments { segments, segment_len });
        }

        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let window: Vec<T> = (0..segment_len)
            .map(|i| T::from(0.5).unwrap() * (T::one() - (two_pi * T::from(i).unwrap() / T::from(segment_len - 1).unwrap()).cos()))
            .collect();
        let fft = FftPlanner::new().plan_fft_forward(segment_len);
        let half = segment_len / 2;
        let zero = Complex::new(T::zero(), T::zero());
        let mut spectra = Spectra {
            ee: vec![T::zero(); half + 1],
            uu: vec![T::zero(); half + 1],
            yy: vec![T::zero(); half + 1],
            eu: vec![zero; half + 1],
            yu: vec![zero; half + 1],
            segments,
            effective_segments: T::zero(),
        };
        let transformed = |x: &[T]| {
            let mean = x.iter().fold(T::zero(), |acc, &v| acc + v) / T::from(x.len()).unwrap();
            let mut buffer: Vec<Complex<T>> = x.iter().zip(&window).map(|(&v, &w)| Complex::new((v - mean) * w, T::zero())).collect();
            fft.process(&mut buffer);
            buffer
        };
        for l in 0..segments {
            let range = l * step..l * step + segment_len;
            let (eb, ub, yb) = (transformed(&e[range.clone()]), transformed(&u[range.clone()]), transformed(&y[range]));
            for f in 0..=half {
                spectra.ee[f] = spectra.ee[f] + eb[f].norm_sqr();
                spectra.uu[f] = spectra.uu[f] + ub[f].norm_sqr();
                spectra.yy[f] = spectra.yy[f] + yb[f].norm_sqr();
                spectra.eu[f] = spectra.eu[f] + eb[f] * ub[f].conj();
                spectra.yu[f] = spectra.yu[f] + yb[f] * ub[f].conj();
            }
        }
        let k = T::from(segments).unwrap();
        for f in 0..=half {
            spectra.ee[f] = spectra.ee[f] / k;
            spectra.uu[f] = spectra.uu[f] / k;
            spectra.yy[f] = spectra.yy[f] / k;
            spectra.eu[f] = spectra.eu[f] / k;
            spectra.yu[f] = spectra.yu[f] / k;
        }

        // Equivalent number of independent segments
        let energy = window.iter().fold(T::zero(), |acc, &w| acc + w * w);
        let correlation = (1..segments).take_while(|m| m * step < segment_len).fold(T::zero(), |acc, m| {
            let c = (0..segment_len - m * step).fold(T::zero(), |acc, t| acc + window[t] * window[t + m * step]) / energy;
            acc + (T::one() - T::from(m).unwrap() / k) * c * c
        });
        spectra.effective_segments = k / (T::one() + T::from(2).unwrap() * correlation);
        Ok(spectra)
    }
}

/// Mean spectra of `Validation::spectra` (windowed DFT units).
struct Spectra<T> {
    ee: Vec<T>,
    uu: Vec<T>,
    yy: Vec<T>,
    eu: Vec<Complex<T>>,
    yu: Vec<Complex<T>>,
    segments: usize,
    effective_segments: T,
}

/// Result of `Validation::frequency_response`.
#[derive(Clone, Debug)]
pub struct FrequencyResponseComparison<T> {
    /// Frequency bins `f` (frequency `f / (segment_len ts)`).
    pub bins: Vec<usize>,
    /// Nonparametric estimate `G(f) = S_yu / S_uu` from the data.
    pub measured: Vec<Complex<T>>,
    /// Response of the model `Ĝ(f) = S_ŷu / S_uu`, estimated with the same segments.
    pub model: Vec<Complex<T>>,
    /// Coherence `γ²_yu(f)` of the output and the input (1 without noise and nonlinearity).
    pub coherence: Vec<T>,
    /// Standard deviation `σ(f)` of the measured response (complex, `E|G - G0|^2 = σ^2`).
    pub stdev: Vec<T>,
    /// Input power at each bin, for `excited`.
    pub input_power: Vec<T>,
    /// Segment length \[samples\].
    pub segment_len: usize,
    /// Equivalent number of independent segments `L`.
    pub effective_segments: T,
}

impl<T: Float> FrequencyResponseComparison<T> {
    /// Frequencies \[Hz\] of the bins for the sampling period `ts`.
    pub fn frequencies(&self, ts: T) -> Vec<T> {
        let df = T::one() / (T::from(self.segment_len).unwrap() * ts);
        self.bins.iter().map(|&f| T::from(f).unwrap() * df).collect()
    }

    /// The bins where the input power is at least `relative` times its maximum: elsewhere the
    /// measured response is not defined by the data.
    pub fn excited(&self, relative: T) -> Self {
        let threshold = relative * self.input_power.iter().fold(T::zero(), |acc, &p| acc.max(p));
        let keep: Vec<usize> = (0..self.bins.len()).filter(|&i| self.input_power[i] >= threshold).collect();
        Self {
            bins: keep.iter().map(|&i| self.bins[i]).collect(),
            measured: keep.iter().map(|&i| self.measured[i]).collect(),
            model: keep.iter().map(|&i| self.model[i]).collect(),
            coherence: keep.iter().map(|&i| self.coherence[i]).collect(),
            stdev: keep.iter().map(|&i| self.stdev[i]).collect(),
            input_power: keep.iter().map(|&i| self.input_power[i]).collect(),
            ..self.clone()
        }
    }

    /// Relative error `|Ĝ - G| / |G|` at each bin.
    pub fn relative_error(&self) -> Vec<T> {
        self.model.iter().zip(&self.measured).map(|(m, g)| (m - g).norm() / g.norm()).collect()
    }

    /// RMS of the relative error over the bins.
    pub fn rms_relative_error(&self) -> T {
        let e = self.relative_error();
        (e.iter().fold(T::zero(), |acc, &v| acc + v * v) / T::from(e.len().max(1)).unwrap()).sqrt()
    }

    /// Gain error `20 log10 |Ĝ / G|` \[dB\] at each bin.
    pub fn gain_error_db(&self) -> Vec<T> {
        let twenty = T::from(20).unwrap();
        self.model.iter().zip(&self.measured).map(|(m, g)| twenty * (m.norm() / g.norm()).log10()).collect()
    }

    /// Phase error `arg(Ĝ / G)` \[rad\], in `(-π, π]`, at each bin.
    pub fn phase_error(&self) -> Vec<T> {
        self.model.iter().zip(&self.measured).map(|(m, g)| (m / g).arg()).collect()
    }

    /// Model error in units of the uncertainty of the data, `|Ĝ - G|^2 / σ^2` at each bin: about
    /// `χ²(2) / 2` (mean 1) for a right model; much larger means a model error beyond the noise.
    pub fn normalized_error(&self) -> Vec<T> {
        self.model.iter().zip(&self.measured).zip(&self.stdev).map(|((m, g), s)| (m - g).norm_sqr() / (*s * *s)).collect()
    }

    /// Mean of `normalized_error` over the bins (about 1 for a right model).
    pub fn mean_normalized_error(&self) -> T {
        let e = self.normalized_error();
        e.iter().fold(T::zero(), |acc, &v| acc + v) / T::from(e.len().max(1)).unwrap()
    }
}

/// Result of `Validation::coherence_test`.
#[derive(Clone, Debug)]
pub struct CoherenceTest<T> {
    /// Frequency bins `f` (frequency `f / (segment_len ts)`).
    pub bins: Vec<usize>,
    /// Coherence `γ²(f)` of the residual and the input at each bin.
    pub coherence: Vec<T>,
    /// Input power at each bin (mean `|U_l(f)|^2` over the segments, windowed), for `excited`.
    pub input_power: Vec<T>,
    /// Segment length \[samples\].
    pub segment_len: usize,
    /// Number of (overlapping) segments `K`.
    pub segments: usize,
    /// Equivalent number of independent segments `L`.
    pub effective_segments: T,
    /// Bound of `γ²(f)` at the given confidence.
    pub bound: T,
}

impl<T: Float> CoherenceTest<T> {
    /// Frequencies \[Hz\] of the bins for the sampling period `ts`.
    pub fn frequencies(&self, ts: T) -> Vec<T> {
        let df = T::one() / (T::from(self.segment_len).unwrap() * ts);
        self.bins.iter().map(|&f| T::from(f).unwrap() * df).collect()
    }

    /// The bins where the input power is at least `relative` times its maximum (e.g. `1e-2` for
    /// -20 dB): elsewhere the test has no power to detect a model error.
    pub fn excited(&self, relative: T) -> Self {
        let threshold = relative * self.input_power.iter().fold(T::zero(), |acc, &p| acc.max(p));
        let keep: Vec<usize> = (0..self.bins.len()).filter(|&i| self.input_power[i] >= threshold).collect();
        Self {
            bins: keep.iter().map(|&i| self.bins[i]).collect(),
            coherence: keep.iter().map(|&i| self.coherence[i]).collect(),
            input_power: keep.iter().map(|&i| self.input_power[i]).collect(),
            ..self.clone()
        }
    }

    /// Bins with `γ²(f) > bound`.
    pub fn outside(&self) -> Vec<usize> {
        self.bins.iter().zip(&self.coherence).filter(|(_, c)| **c > self.bound).map(|(&f, _)| f).collect()
    }

    /// Fraction of the bins with `γ²(f) > bound`; about `1 - confidence` for a right model.
    pub fn fraction_outside(&self) -> T {
        T::from(self.outside().len()).unwrap() / T::from(self.bins.len().max(1)).unwrap()
    }

    /// Mean of `γ²(f)` over the bins, to compare with `expected_coherence`.
    pub fn mean_coherence(&self) -> T {
        self.coherence.iter().fold(T::zero(), |acc, &c| acc + c) / T::from(self.bins.len().max(1)).unwrap()
    }

    /// Expectation `1 / L` of `γ²(f)` for a right model.
    pub fn expected_coherence(&self) -> T {
        T::one() / self.effective_segments
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

/// Result of `Validation::autocorrelation`.
#[derive(Clone, Debug)]
pub struct WhitenessTest<T> {
    /// Lags `τ = 1 ..= max_lag` \[samples\].
    pub lags: Vec<usize>,
    /// Normalized autocorrelation `r(τ)` of the residual at each lag.
    pub correlation: Vec<T>,
    /// Bound `z / sqrt(N)` of each `r(τ)` for a white residual.
    pub bound: T,
    /// Ljung-Box statistic `Q = N (N + 2) Σ r(τ)^2 / (N - τ)`, `χ²(max_lag)` for a white residual.
    pub ljung_box: T,
}

impl<T: Float> WhitenessTest<T> {
    /// Lags with `|r(τ)| > bound`.
    pub fn outside(&self) -> Vec<usize> {
        self.lags.iter().zip(&self.correlation).filter(|(_, r)| r.abs() > self.bound).map(|(&tau, _)| tau).collect()
    }

    /// Fraction of the lags with `|r(τ)| > bound`; about the nominal level for a white residual.
    pub fn fraction_outside(&self) -> T {
        T::from(self.outside().len()).unwrap() / T::from(self.lags.len().max(1)).unwrap()
    }

    /// `max |r(τ)| / bound` over the lags.
    pub fn max_ratio(&self) -> T {
        self.correlation.iter().fold(T::zero(), |acc, r| acc.max(r.abs())) / self.bound
    }

    /// Upper quantile of `χ²(max_lag)` at the normal quantile `z` (one-sided: `z = 2.33` for 99 %,
    /// `1.64` for 95 %), by the Wilson-Hilferty approximation
    /// `m (1 - 2/(9m) + z sqrt(2/(9m)))^3`: the residual is white at that level if
    /// `ljung_box <= ljung_box_bound(z)`.
    pub fn ljung_box_bound(&self, z: T) -> T {
        let m = T::from(self.lags.len()).unwrap();
        let c = T::from(2).unwrap() / (T::from(9).unwrap() * m);
        m * (T::one() - c + z * c.sqrt()).powi(3)
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
