//! Frequency-domain sample maximum likelihood (SML) estimator of Pintelon and Schoukens.
//!
//! The input and output of a periodic excitation are measured over `M >= 4` periods. At every
//! excited line the DFT of each period gives the sample means `Ū`, `Ȳ` and the sample
//! (co)variances of the means `σ̂²_U`, `σ̂²_Y`, `σ̂_YU` (a nonparametric noise model, so that no noise
//! model has to be identified). The SML estimate of `G = B / A` minimizes
//!
//! ```text
//! V(θ) = Σ_k |B(Ω_k, θ) Ū(k) - A(Ω_k, θ) Ȳ(k)|^2 / D_k(θ)
//! D_k(θ) = σ̂²_Y(k) |A|^2 + σ̂²_U(k) |B|^2 - 2 Re(σ̂_YU(k) A B̄)
//! ```
//!
//! i.e. the equation error weighted by its variance: the maximum-likelihood cost with the noise
//! (co)variances replaced by their sample estimates. It is consistent for noise on both the input
//! and the output (errors-in-variables) with `M >= 4` periods, and nearly as efficient as the ML
//! estimator with known noise (covariance larger by `(M - 2) / (M - 3)`). Periodic excitation in
//! steady state (the transient periods left out) has no leakage, so the model is that of the lines
//! alone.
//!
//! References: J. Schoukens, R. Pintelon, Y. Rolain, "Frequency-domain system identification
//! using non-parametric noise models estimated from a small number of data sets", Automatica 33(6),
//! 1997; R. Pintelon, J. Schoukens, "System Identification: A Frequency Domain Approach", 2nd ed.,
//! Wiley, 2012, Ch. 10.

use nalgebra::{Complex, DMatrix, DVector, RealField};
use num_traits::Float;
use rustfft::{FftNum, FftPlanner};
use thiserror::Error;

use crate::{Continuous, Discrete, Polynomial, TransferFunction, TransferFunctionWithDelay};

/// Errors of the SML estimator.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SmlError {
    /// The input and output have different lengths.
    #[error("input and output lengths differ: {input} vs {output}")]
    LengthMismatch { input: usize, output: usize },
    /// The record is not a whole number of periods.
    #[error("{samples} samples are not a whole number of periods of {period} samples")]
    NotWholePeriods { samples: usize, period: usize },
    /// Fewer than 4 periods: the SML cost has no finite expected value.
    #[error("{periods} periods, at least 4 needed")]
    TooFewPeriods { periods: usize },
    /// A line outside `0 ..= period / 2`.
    #[error("line {line} outside 0 ..= {} for the period of {period} samples", period / 2)]
    InvalidLine { line: usize, period: usize },
    /// A spectral line with non-finite values, a negative variance, or no noise on the input and
    /// the output (`σ̂²_U = σ̂²_Y = 0`, e.g. noise-free data: the cost is not defined).
    #[error("spectral line {index}: non-finite values, negative variance or no noise")]
    InvalidSpectralLine { index: usize },
    /// Fewer real equations `2 F` than the parameters `nθ` plus one.
    #[error("{lines} lines for {parameters} parameters (2 F > nθ needed)")]
    TooFewLines { lines: usize, parameters: usize },
    /// The sampling period is not positive and finite.
    #[error("invalid sampling period")]
    InvalidSamplingPeriod,
    /// A least-squares problem is singular, not finite or its SVD does not converge, or no initial
    /// model has a finite cost.
    #[error("singular least-squares problem (check orders and data)")]
    Singular,
}

/// Sample means and (co)variances of the means of the input and output spectra at one line.
#[derive(Clone, Copy, Debug)]
pub struct SpectralLine<T> {
    /// Angular frequency \[rad/s\].
    pub omega: T,
    /// Mean input spectrum `Ū = 1/M Σ_m U^[m]`.
    pub u: Complex<T>,
    /// Mean output spectrum `Ȳ = 1/M Σ_m Y^[m]`.
    pub y: Complex<T>,
    /// Variance of `Ū`: `σ̂²_U = 1/(M (M - 1)) Σ_m |U^[m] - Ū|^2`.
    pub var_u: T,
    /// Variance of `Ȳ`: `σ̂²_Y = 1/(M (M - 1)) Σ_m |Y^[m] - Ȳ|^2`.
    pub var_y: T,
    /// Covariance of `Ȳ` and `Ū`: `σ̂_YU = 1/(M (M - 1)) Σ_m (Y^[m] - Ȳ) (U^[m] - Ū)*`.
    pub cov_yu: Complex<T>,
}

impl<T: Float> SpectralLine<T> {
    /// The line with `Ū` multiplied by `h` and `Ȳ` by `c`, and the (co)variances accordingly:
    /// `σ̂²_U |h|^2`, `σ̂²_Y |c|^2`, `σ̂_YU c h̄`. E.g. `h` a known input filter (the zero-order hold,
    /// see `SmlData::compensate_zero_order_hold`), `c` a known factor of the model moved to the
    /// output (see `identify_continuous_with_delay`).
    pub fn scaled(&self, h: Complex<T>, c: Complex<T>) -> Self {
        Self {
            omega: self.omega,
            u: self.u * h,
            y: self.y * c,
            var_u: self.var_u * h.norm_sqr(),
            var_y: self.var_y * c.norm_sqr(),
            cov_yu: self.cov_yu * c * h.conj(),
        }
    }

    /// Measured response `Ĝ = Ȳ / Ū`.
    pub fn response(&self) -> Complex<T> {
        self.y / self.u
    }

    /// Standard deviation of `Ĝ` to first order,
    /// `sqrt(max(0, (σ̂²_Y + |Ĝ|^2 σ̂²_U - 2 Re(Ĝ̄ σ̂_YU)) / |Ū|^2))`.
    pub fn response_std(&self) -> T {
        let g = self.response();
        let two = T::one() + T::one();
        let var = (self.var_y + g.norm_sqr() * self.var_u - two * (g.conj() * self.cov_yu).re) / self.u.norm_sqr();
        Float::sqrt(Float::max(var, T::zero()))
    }
}

/// Spectral lines of `M` periods, the data of the SML estimator.
#[derive(Clone, Debug)]
pub struct SmlData<T> {
    /// The lines used in the cost (the excited ones).
    pub lines: Vec<SpectralLine<T>>,
    /// Number of periods `M` the (co)variances are estimated from.
    pub periods: usize,
}

impl<T: Float> SmlData<T> {
    /// Data from lines computed elsewhere, checked: `M >= 4` periods, and at every line finite
    /// values, nonnegative variances and noise on the input or the output.
    pub fn new(lines: Vec<SpectralLine<T>>, periods: usize) -> Result<Self, SmlError> {
        if periods < 4 {
            return Err(SmlError::TooFewPeriods { periods });
        }
        for (index, l) in lines.iter().enumerate() {
            let finite = [l.omega, l.u.re, l.u.im, l.y.re, l.y.im, l.var_u, l.var_y, l.cov_yu.re, l.cov_yu.im].iter().all(|v| v.is_finite());
            if !finite || l.var_u < T::zero() || l.var_y < T::zero() || l.var_u + l.var_y == T::zero() {
                return Err(SmlError::InvalidSpectralLine { index });
            }
        }
        Ok(Self { lines, periods })
    }

    /// Lines of the periodic input / output records `u`, `y` (sampling period `ts` \[s\]) of
    /// `period` samples per period: the DFT `X^[m](k) = Σ_t x[m N + t] e^(-j 2π k t / N)` of each
    /// period `m` at the `lines` `k` (`ω = 2π k / (N ts)`), their means and the (co)variances of
    /// the means. The records must be whole periods in steady state (leave out the transient
    /// periods), at least 4.
    pub fn from_periods(u: &[T], y: &[T], ts: T, period: usize, lines: &[usize]) -> Result<Self, SmlError>
    where
        T: FftNum,
    {
        if u.len() != y.len() {
            return Err(SmlError::LengthMismatch { input: u.len(), output: y.len() });
        }
        if !(ts > T::zero() && ts.is_finite()) {
            return Err(SmlError::InvalidSamplingPeriod);
        }
        if period == 0 || !u.len().is_multiple_of(period) {
            return Err(SmlError::NotWholePeriods { samples: u.len(), period });
        }
        let periods = u.len() / period;
        if periods < 4 {
            return Err(SmlError::TooFewPeriods { periods });
        }
        if let Some(&line) = lines.iter().find(|&&k| k > period / 2) {
            return Err(SmlError::InvalidLine { line, period });
        }

        let fft = FftPlanner::new().plan_fft_forward(period);
        let spectra = |x: &[T]| -> Vec<Vec<Complex<T>>> {
            x.chunks(period)
                .map(|chunk| {
                    let mut buffer: Vec<Complex<T>> = chunk.iter().map(|&v| Complex::new(v, T::zero())).collect();
                    fft.process(&mut buffer);
                    lines.iter().map(|&k| buffer[k]).collect()
                })
                .collect()
        };
        let (us, ys) = (spectra(u), spectra(y));
        let m = T::from(periods).unwrap();
        let scale = T::one() / (m * (m - T::one()));
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let zero = Complex::new(T::zero(), T::zero());
        let lines = lines
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let u_mean = us.iter().fold(zero, |acc, s| acc + s[i]) / m;
                let y_mean = ys.iter().fold(zero, |acc, s| acc + s[i]) / m;
                let (mut var_u, mut var_y, mut cov_yu) = (T::zero(), T::zero(), zero);
                for (su, sy) in us.iter().zip(&ys) {
                    let (du, dy) = (su[i] - u_mean, sy[i] - y_mean);
                    var_u = var_u + du.norm_sqr();
                    var_y = var_y + dy.norm_sqr();
                    cov_yu = cov_yu + dy * du.conj();
                }
                SpectralLine {
                    omega: two_pi * T::from(k).unwrap() / (T::from(period).unwrap() * ts),
                    u: u_mean,
                    y: y_mean,
                    var_u: var_u * scale,
                    var_y: var_y * scale,
                    cov_yu: cov_yu * scale,
                }
            })
            .collect();
        Self::new(lines, periods)
    }

    /// `from_periods` on a raw record: the first `transient_periods` periods (the transient from
    /// rest) and the incomplete period at the end are left out. At least 4 whole periods must
    /// remain (`TooFewPeriods` otherwise).
    pub fn from_record(u: &[T], y: &[T], ts: T, period: usize, transient_periods: usize, lines: &[usize]) -> Result<Self, SmlError>
    where
        T: FftNum,
    {
        if u.len() != y.len() {
            return Err(SmlError::LengthMismatch { input: u.len(), output: y.len() });
        }
        if period == 0 {
            return Err(SmlError::NotWholePeriods { samples: u.len(), period });
        }
        let periods = (u.len() / period).saturating_sub(transient_periods);
        if periods < 4 {
            return Err(SmlError::TooFewPeriods { periods });
        }
        let range = transient_periods * period..(transient_periods + periods) * period;
        Self::from_periods(&u[range.clone()], &y[range], ts, period, lines)
    }

    /// The lines with `Ū` multiplied by the zero-order hold `h = e^(-jω ts / 2) sinc(ω ts / 2)`
    /// (`sinc x = sin x / x`) of the sampling period `ts` (see `SpectralLine::scaled`): the data of
    /// a held input, fitted by a continuous-time model `G(s)` of the plant behind the hold, as
    /// `Y = G(jω) h Ū` (up to aliasing, small where `|G|` rolls off well below the Nyquist
    /// frequency).
    pub fn compensate_zero_order_hold(&self, ts: T) -> Self {
        let two = T::one() + T::one();
        let lines = self
            .lines
            .iter()
            .map(|l| {
                let x = l.omega * ts / two;
                let sinc = if x == T::zero() { T::one() } else { Float::sin(x) / x };
                l.scaled(Complex::new(T::zero(), -x).exp() * sinc, Complex::new(T::one(), T::zero()))
            })
            .collect();
        Self { lines, periods: self.periods }
    }

    /// The lines of frequency `ω / 2π` in `[low, high]` \[Hz\].
    pub fn in_band(&self, low: T, high: T) -> Self {
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let lines = self.lines.iter().filter(|l| (low..=high).contains(&(l.omega / two_pi))).copied().collect();
        Self { lines, periods: self.periods }
    }
}

/// Options of `excited_lines`.
#[derive(Clone, Debug)]
pub struct ExcitationOptions<T> {
    /// Lowest power `|Ū|^2` of an excited line relative to the highest (DC left out), above the
    /// rounding of a computed input. Small (default 1e-12), so that steeply shaped amplitudes
    /// (e.g. `∝ k^2`, for a flat output of a double integrator) keep their low lines: a fraction of
    /// the maximum would drop them, and with them the rigid-body mode.
    pub floor: T,
    /// Lowest `|Ū|^2 / σ̂²_U` of an excited line (default 100). At a line not excited, of a noisy
    /// input, the ratio follows `F(2, 2(M - 1))`: above 100 with probability
    /// `(1 + 100 / (M - 1))^-(M - 1)`, 2.5e-5 for `M = 4`.
    pub snr: T,
}

impl<T: Float> Default for ExcitationOptions<T> {
    fn default() -> Self {
        Self { floor: T::from(1e-12).unwrap(), snr: T::from(100).unwrap() }
    }
}

/// Excited lines `k` (`1 ..= period / 2`, DC left out) of the periodic input `u` (`M` whole periods
/// of `period` samples in steady state, the transient left out; an incomplete period at the end is
/// ignored), to pass to `SmlData::from_periods`. From the DFT `U^[m](k)` of each period, `Ū` and
/// `σ̂²_U = 1/(M (M - 1)) Σ_m |U^[m] - Ū|^2` (as `SmlData::from_periods`), a line is excited if
///
/// ```text
/// |Ū(k)|^2 >= floor · max_(k' >= 1) |Ū(k')|^2   and   |Ū(k)|^2 >= snr · σ̂²_U(k)
/// ```
///
/// (see `ExcitationOptions`). Empty for fewer than 2 periods or a zero `period`.
pub fn excited_lines<T: Float + FftNum>(u: &[T], period: usize, options: &ExcitationOptions<T>) -> Vec<usize> {
    if period == 0 || u.len() / period < 2 {
        return Vec::new();
    }
    let periods = u.len() / period;
    let fft = FftPlanner::new().plan_fft_forward(period);
    let half = period / 2;
    let zero = Complex::new(T::zero(), T::zero());
    let spectra: Vec<Vec<Complex<T>>> = u[..periods * period]
        .chunks(period)
        .map(|chunk| {
            let mut buffer: Vec<Complex<T>> = chunk.iter().map(|&v| Complex::new(v, T::zero())).collect();
            fft.process(&mut buffer);
            buffer.truncate(half + 1);
            buffer
        })
        .collect();
    let m = T::from(periods).unwrap();
    let (power, variance): (Vec<T>, Vec<T>) = (0..=half)
        .map(|k| {
            let mean = spectra.iter().fold(zero, |acc, s| acc + s[k]) / m;
            let var = spectra.iter().fold(T::zero(), |acc, s| acc + (s[k] - mean).norm_sqr()) / (m * (m - T::one()));
            (mean.norm_sqr(), var)
        })
        .unzip();
    let max = power[1..].iter().fold(T::zero(), |acc, &p| Float::max(acc, p));
    (1..=half).filter(|&k| power[k] > T::zero() && power[k] >= options.floor * max && power[k] >= options.snr * variance[k]).collect()
}

/// Options of the SML estimator.
#[derive(Clone, Debug)]
pub struct SmlOptions<T> {
    /// Maximum number of iterations of the weighted linear least squares giving the starting
    /// point (IQML, see `identify_continuous`).
    pub initial_iterations: usize,
    /// Maximum number of Levenberg-Marquardt iterations (accepted steps).
    pub max_iterations: usize,
    /// Convergence threshold of the relative parameter change `‖Δθ‖ / ‖θ‖`.
    pub tolerance: T,
}

impl<T: Float> Default for SmlOptions<T> {
    fn default() -> Self {
        Self { initial_iterations: 20, max_iterations: 100, tolerance: T::from(1e-10).unwrap() }
    }
}

/// Why the Levenberg-Marquardt iterations stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// The relative parameter change of an accepted step fell below `tolerance`.
    Converged,
    /// No step decreased the cost up to `μ > 1e10`: the cost cannot be decreased further within
    /// rounding (e.g. at a minimum reached before `tolerance`), but this is not a guarantee of one.
    Stalled,
    /// `max_iterations` steps were accepted without convergence.
    MaxIterations,
}

/// Result of the SML estimator.
#[derive(Clone, Debug)]
pub struct SmlResult<T, D> {
    /// Identified model `B / A` (not reduced, so that the coefficients match the standard
    /// deviations): monic `A` in `s` for `identify_continuous`; for `identify_discrete`, as
    /// polynomials in `z` like `Arx::transfer_function`, with leading coefficient 1 of `A`.
    pub model: TransferFunction<T, D>,
    /// Standard deviations of the numerator coefficients of `model` (Cramér-Rao bound of the
    /// sample noise model, `(M - 2) / (M - 3)` times that of known noise); 0 for the fixed ones.
    pub std_numerator: Vec<T>,
    /// Standard deviations of the denominator coefficients of `model`; 0 for the fixed ones.
    pub std_denominator: Vec<T>,
    /// SML cost `V(θ̂)` at the estimate.
    pub cost: T,
    /// Expected cost of a model without error `(M - 1) / (M - 2) (F - nθ / 2)` (`F` lines, `nθ` free
    /// parameters, circular complex Gaussian noise).
    pub expected_cost: T,
    /// Standard deviation of the cost `√F (M - 1) / (M - 2) √((M - 1) / (M - 3))`: a cost above
    /// `expected_cost + 2 cost_std` or so points to a model error (order too low, delay, leakage).
    pub cost_std: T,
    /// Number of Levenberg-Marquardt iterations done.
    pub iterations: usize,
    /// Why the iterations stopped.
    pub termination: Termination,
    /// Number of lines `F` of the data.
    pub lines: usize,
    /// Number of periods `M` of the data.
    pub periods: usize,
    /// Number of free parameters `nθ = n + m + 1`.
    pub parameters: usize,
}

impl<T, D> SmlResult<T, D> {
    /// Whether the relative parameter change fell below `tolerance` (`Termination::Converged`).
    pub fn converged(&self) -> bool {
        self.termination == Termination::Converged
    }
}

impl<T: Float, D> SmlResult<T, D> {
    /// Cost test `cost <= expected_cost + sigmas · cost_std` (e.g. `sigmas = 2`): `false` points to
    /// a model error (order too low, delay, leakage). To choose a structure, prefer `bic`: with few
    /// periods or a low signal-to-noise ratio the test is loose, and passes structures with fewer
    /// poles and a much higher cost.
    pub fn cost_test(&self, sigmas: T) -> bool {
        self.cost <= self.expected_cost + sigmas * self.cost_std
    }

    /// Bayesian information criterion of the SML cost, `2V / r + nθ ln(2F)` with
    /// `r = (M - 1) / (M - 2)`: `2V / r` is `-2 ln L` (up to a constant) of the residuals `e_k`, as
    /// circular complex Gaussian of variance about `r` (`2F` real equations). The smaller the
    /// better; it compares structures (orders, integrators, delay) on the same data.
    ///
    /// Not `Validation::bic` (`N ln V + p ln N` of the time-domain residuals): that one has the
    /// noise variance unknown and estimated from the residuals, this one has it known from the
    /// periods, so the cost enters linearly.
    ///
    /// Prefer it to `cost_test` to choose a structure: with few periods or a low signal-to-noise
    /// ratio the cost test is loose, and passes structures with fewer poles and a much higher cost.
    pub fn bic(&self) -> T {
        information_criterion(self.cost, self.periods, self.parameters, Float::ln(T::from(2 * self.lines).unwrap()))
    }

    /// Akaike information criterion of the SML cost, `2V / r + 2 nθ` (see `bic`, of which it has the
    /// smaller penalty for more than 3 lines; not `Validation::aic`).
    pub fn aic(&self) -> T {
        information_criterion(self.cost, self.periods, self.parameters, T::from(2).unwrap())
    }
}

/// `2V / r + nθ penalty`, `r = (M - 1) / (M - 2)`.
fn information_criterion<T: Float>(cost: T, periods: usize, parameters: usize, penalty: T) -> T {
    let m = T::from(periods).unwrap();
    let two = T::from(2).unwrap();
    let r = (m - T::one()) / (m - two);
    two * cost / r + T::from(parameters).unwrap() * penalty
}

/// Result of `identify_continuous_with_delay`.
#[derive(Clone, Debug)]
pub struct SmlResultWithDelay<T> {
    /// Identified model `e^(-delay s) B(s) / (s^q A(s))`, `A` monic; `model.tf.denominator`
    /// includes `s^q`.
    pub model: TransferFunctionWithDelay<T>,
    /// The SML result of the rational part `B / (s^q A)`; its `std_denominator` is 0 for the fixed
    /// coefficients (the leading 1 and those of the integrators), and `parameters` is `n + m + 1`.
    pub result: SmlResult<T, Continuous>,
}

/// Cost of each line `|e_k|^2 = |B Ū - A Ȳ|^2 / D_k`,
/// `D_k = σ̂²_Y |A|^2 + σ̂²_U |B|^2 - 2 Re(σ̂_YU A B̄)`, for the continuous-time `model = B / A`
/// (infinite where `D_k` is not positive). Their sum is the SML cost; each has the expected value
/// of about `expected_cost / F`, so lines far above point to a local model error (or a
/// disturbance).
pub fn line_costs<T: Float>(data: &SmlData<T>, model: &TransferFunction<T, Continuous>) -> Vec<T> {
    line_costs_of(data, |omega| {
        let s = Complex::new(T::zero(), omega);
        (evaluate(&model.numerator, s), evaluate(&model.denominator, s))
    })
}

/// `line_costs` of the discrete-time `model` (polynomials in `z = e^(jω ts)`).
pub fn line_costs_discrete<T: Float>(data: &SmlData<T>, ts: T, model: &TransferFunction<T, Discrete>) -> Vec<T> {
    line_costs_of(data, |omega| {
        let z = Complex::new(T::zero(), omega * ts).exp();
        (evaluate(&model.numerator, z), evaluate(&model.denominator, z))
    })
}

/// `line_costs` of `model = e^(-delay s) B / A`, as `B e^(-jω delay) / A`: on the data passed to
/// `identify_continuous_with_delay`, their sum is its cost.
pub fn line_costs_with_delay<T: Float>(data: &SmlData<T>, model: &TransferFunctionWithDelay<T>) -> Vec<T> {
    line_costs_of(data, |omega| {
        let s = Complex::new(T::zero(), omega);
        ((s * -model.delay).exp() * evaluate(&model.tf.numerator, s), evaluate(&model.tf.denominator, s))
    })
}

/// Polynomial of descending coefficients at `x`.
fn evaluate<T: Float>(p: &Polynomial<T>, x: Complex<T>) -> Complex<T> {
    p.iter().fold(Complex::new(T::zero(), T::zero()), |acc, &c| acc * x + c)
}

/// `|B Ū - A Ȳ|^2 / D_k` with `(B, A) = polynomials(ω)`.
fn line_costs_of<T: Float>(data: &SmlData<T>, polynomials: impl Fn(T) -> (Complex<T>, Complex<T>)) -> Vec<T> {
    data.lines
        .iter()
        .map(|l| {
            let (b, a) = polynomials(l.omega);
            let d = variance(l, a, b);
            if d > T::zero() { (b * l.u - a * l.y).norm_sqr() / d } else { T::infinity() }
        })
        .collect()
}

/// SML estimate of the continuous-time model
///
/// ```text
/// G(s) = B(s) / A(s) = (b_m s^m + ... + b_0) / (s^n + a_(n-1) s^(n-1) + ... + a_0)
/// ```
///
/// with `n = denom_order`, `m = numer_order` (argument order as in `levy::identify`).
///
/// The polynomials are in the scaled variable `s / ω_s`, `ω_s = √(ω_min ω_max)` of the nonzero line
/// frequencies, for the conditioning; the result is in `s`. The steps:
///
/// 1. Starting point by iterative quadratic maximum likelihood (IQML): weighted linear least
///    squares on the equation error `B Ū - A Ȳ`, weighted by `1 / D_k` of the previous estimate
///    (of `A = 1`, `B = Ȳ / Ū` at first, a Sanathanan-Koerner iteration with the noise weighting),
///    up to `initial_iterations` times; the estimate of the lowest SML cost is kept.
/// 2. Levenberg-Marquardt on `V(θ)` (residuals `(B Ū - A Ȳ) / √D_k` with the analytic Jacobian,
///    Marquardt scaling), from the starting point.
///
/// The model is that of the lines: the data of a zero-order-hold input are fitted by a
/// continuous-time model only in the band where the hold is negligible (or use
/// `identify_discrete`).
///
/// ```
/// use dsmc::system_identification::frequency_response::sml::{identify_continuous, SmlData, SmlOptions, SpectralLine};
/// use num_complex::Complex64;
///
/// // G(s) = 100 / (s^2 + 2 s + 100) at 40 lines, with small noise of known (co)variance
/// let g = |w: f64| Complex64::new(100.0, 0.0) / (Complex64::new(100.0 - w * w, 2.0 * w));
/// let lines: Vec<SpectralLine<f64>> = (1..=40)
///     .map(|k| {
///         let omega = 0.5 * k as f64;
///         let noise = Complex64::from_polar(1e-3, k as f64);
///         let u = Complex64::new(1.0, 0.0);
///         SpectralLine { omega, u, y: g(omega) * u + noise, var_u: 0.0, var_y: 1e-6, cov_yu: Complex64::new(0.0, 0.0) }
///     })
///     .collect();
/// let data = SmlData::new(lines, 8).unwrap();
/// let result = identify_continuous(&data, 2, 0, &SmlOptions::default()).unwrap();
/// let a = &result.model.denominator;
/// assert!((a[1] - 2.0).abs() < 0.01 && (a[2] - 100.0).abs() < 0.1, "{:?}", result.model);
/// ```
pub fn identify_continuous<T: Float + RealField>(
    data: &SmlData<T>,
    denom_order: usize,
    numer_order: usize,
    options: &SmlOptions<T>,
) -> Result<SmlResult<T, Continuous>, SmlError> {
    let (n, m) = (denom_order, numer_order);
    let positive = data.lines.iter().map(|l| Float::abs(l.omega)).filter(|&w| w > T::zero());
    let (low, high) = positive.fold((T::infinity(), T::zero()), |(lo, hi), w| (Float::min(lo, w), Float::max(hi, w)));
    let omega_s = if high > T::zero() { Float::sqrt(low * high) } else { T::one() };
    let basis: Vec<LineBasis<T>> = data
        .lines
        .iter()
        .map(|l| {
            let s = Complex::new(T::zero(), l.omega / omega_s);
            LineBasis { den: powers(s, n + 1), num: powers(s, m + 1) }
        })
        .collect();
    let fit = fit(data, &basis, n, options)?;

    // Coefficient of s^i: a_i / ω_s^i, times ω_s^n for a monic A
    let factor = |i: usize| Float::powi(omega_s, n as i32 - i as i32);
    let (a, b) = (fit.denominator(n, n), &fit.theta[n..]);
    let denominator: Vec<T> = (0..=n).rev().map(|i| a[i] * factor(i)).collect();
    let numerator: Vec<T> = (0..=m).rev().map(|j| b[j] * factor(j)).collect();
    let std_denominator: Vec<T> = (0..=n).rev().map(|i| if i == n { T::zero() } else { fit.std[i] * factor(i) }).collect();
    let std_numerator: Vec<T> = (0..=m).rev().map(|j| fit.std[n + j] * factor(j)).collect();
    Ok(fit.result(data, TransferFunction::from_polynomials(Polynomial(numerator), Polynomial(denominator)), std_numerator, std_denominator))
}

/// SML estimate of the continuous-time model with `q = integrators` integrators and a known dead
/// time `delay` \[s\]
///
/// ```text
/// G(s) = e^(-delay s) B(s) / (s^q A(s)),   A monic of degree n = denom_order, B of degree m = numer_order
/// ```
///
/// by `identify_continuous` of `B / A` on the lines with `Ȳ` multiplied by
/// `c = (jω)^q e^(jω delay)` (`SpectralLine::scaled`; the noise (co)variances likewise). The free
/// parameters remain `nθ = n + m + 1`; `result.model.denominator` includes `s^q` and
/// `result.std_denominator` is 0 for the fixed coefficients (the leading 1 and the `q` trailing
/// zeros). For the data of a zero-order-hold input, apply `SmlData::compensate_zero_order_hold`
/// first. A line at `ω = 0` with integrators has `c = 0`: leave it out (`SmlData::in_band`).
pub fn identify_continuous_with_delay<T: Float + RealField>(
    data: &SmlData<T>,
    denom_order: usize,
    numer_order: usize,
    integrators: usize,
    delay: T,
    options: &SmlOptions<T>,
) -> Result<SmlResultWithDelay<T>, SmlError> {
    let scaled = SmlData {
        lines: data
            .lines
            .iter()
            .map(|l| {
                let c = Complex::new(T::zero(), l.omega).powi(integrators as i32) * Complex::new(T::zero(), l.omega * delay).exp();
                l.scaled(Complex::new(T::one(), T::zero()), c)
            })
            .collect(),
        periods: data.periods,
    };
    let mut result = identify_continuous(&scaled, denom_order, numer_order, options)?;
    let mut denominator = result.model.denominator.0.clone();
    denominator.resize(denominator.len() + integrators, T::zero());
    result.std_denominator.resize(result.std_denominator.len() + integrators, T::zero());
    result.model = TransferFunction::from_polynomials(result.model.numerator.clone(), Polynomial(denominator));
    Ok(SmlResultWithDelay { model: TransferFunctionWithDelay::new(result.model.clone(), delay), result })
}

/// SML estimate of the discrete-time model (sampling period `ts` \[s\], `z = e^(jω ts)`)
///
/// ```text
/// G(z) = z^-nk (b_0 + b_1 z^-1 + ... + b_m z^-m) / (1 + a_1 z^-1 + ... + a_n z^-n)
/// ```
///
/// with `n = denom_order`, `m = numer_order`, `nk = input_delay`, by the steps of
/// `identify_continuous`. The model is returned as polynomials in `z` (both multiplied by
/// `z^max(n, nk + m)`, as `Arx::transfer_function`). Exact for the data of a zero-order-hold
/// input to a continuous-time plant (its zero-order-hold equivalent) of the orders.
pub fn identify_discrete<T: Float + RealField>(
    data: &SmlData<T>,
    ts: T,
    denom_order: usize,
    numer_order: usize,
    input_delay: usize,
    options: &SmlOptions<T>,
) -> Result<SmlResult<T, Discrete>, SmlError> {
    if !(ts > T::zero() && ts.is_finite()) {
        return Err(SmlError::InvalidSamplingPeriod);
    }
    let (n, m, nk) = (denom_order, numer_order, input_delay);
    let basis: Vec<LineBasis<T>> = data
        .lines
        .iter()
        .map(|l| {
            let z_inv = Complex::new(T::zero(), -l.omega * ts).exp();
            let delay = Complex::new(T::zero(), -l.omega * ts * T::from(nk).unwrap()).exp();
            LineBasis { den: powers(z_inv, n + 1), num: powers(z_inv, m + 1).into_iter().map(|p| p * delay).collect() }
        })
        .collect();
    let fit = fit(data, &basis, 0, options)?;

    let order = n.max(nk + m);
    let a = fit.denominator(n, 0);
    let mut denominator = a.clone();
    denominator.resize(order + 1, T::zero());
    let mut std_denominator: Vec<T> = std::iter::once(T::zero()).chain(fit.std[..n].iter().copied()).collect();
    std_denominator.resize(order + 1, T::zero());
    let mut numerator = vec![T::zero(); nk];
    numerator.extend_from_slice(&fit.theta[n..]);
    numerator.resize(order + 1, T::zero());
    let mut std_numerator = vec![T::zero(); nk];
    std_numerator.extend_from_slice(&fit.std[n..]);
    std_numerator.resize(order + 1, T::zero());
    Ok(fit.result(data, TransferFunction::from_polynomials(Polynomial(numerator), Polynomial(denominator)), std_numerator, std_denominator))
}

/// `[1, x, ..., x^(len - 1)]`.
fn powers<T: Float>(x: Complex<T>, len: usize) -> Vec<Complex<T>> {
    std::iter::successors(Some(Complex::new(T::one(), T::zero())), |&p| Some(p * x)).take(len).collect()
}

/// Basis functions of `A` (`den[i]`, `i = 0 ..= n`) and `B` (`num[j]`, `j = 0 ..= m`) at one line:
/// `A = Σ a_i den[i]`, `B = Σ b_j num[j]`.
struct LineBasis<T> {
    den: Vec<Complex<T>>,
    num: Vec<Complex<T>>,
}

/// Parameters `θ = [a_i (i ≠ fixed, ascending), b_0, ..., b_m]` with `a_fixed = 1`.
struct Fit<T> {
    theta: Vec<T>,
    std: Vec<T>,
    cost: T,
    parameters: usize,
    iterations: usize,
    termination: Termination,
}

impl<T: Float> Fit<T> {
    /// `[a_0, ..., a_n]` with `a_fixed = 1`.
    fn denominator(&self, n: usize, fixed: usize) -> Vec<T> {
        let mut free = self.theta[..n].iter();
        (0..=n).map(|i| if i == fixed { T::one() } else { *free.next().unwrap() }).collect()
    }

    fn result<D>(&self, data: &SmlData<T>, model: TransferFunction<T, D>, std_numerator: Vec<T>, std_denominator: Vec<T>) -> SmlResult<T, D> {
        let (f, m) = (T::from(data.lines.len()).unwrap(), T::from(data.periods).unwrap());
        let one = T::one();
        let two = one + one;
        let ratio = (m - one) / (m - two);
        SmlResult {
            model,
            std_numerator,
            std_denominator,
            cost: self.cost,
            expected_cost: ratio * (f - T::from(self.parameters).unwrap() / two),
            cost_std: Float::sqrt(f) * ratio * Float::sqrt((m - one) / (m - T::from(3).unwrap())),
            iterations: self.iterations,
            termination: self.termination,
            lines: data.lines.len(),
            periods: data.periods,
            parameters: self.parameters,
        }
    }
}

/// `A`, `B` of `θ` at a line.
fn polynomials<T: Float>(basis: &LineBasis<T>, theta: &[T], fixed: usize) -> (Complex<T>, Complex<T>) {
    let n = basis.den.len() - 1;
    let mut free = theta[..n].iter();
    let a = basis.den.iter().enumerate().fold(Complex::new(T::zero(), T::zero()), |acc, (i, &d)| {
        acc + d * if i == fixed { T::one() } else { *free.next().unwrap() }
    });
    let b = basis.num.iter().zip(&theta[n..]).fold(Complex::new(T::zero(), T::zero()), |acc, (&p, &c)| acc + p * c);
    (a, b)
}

/// Variance `D = σ̂²_Y |A|^2 + σ̂²_U |B|^2 - 2 Re(σ̂_YU A B̄)` of the equation error `B Ū - A Ȳ`.
fn variance<T: Float>(line: &SpectralLine<T>, a: Complex<T>, b: Complex<T>) -> T {
    let two = T::one() + T::one();
    line.var_y * a.norm_sqr() + line.var_u * b.norm_sqr() - two * (line.cov_yu * a * b.conj()).re
}

/// Residuals `[Re e_k, Im e_k]`, `e_k = (B Ū - A Ȳ) / √D_k`, and optionally their Jacobian; `None`
/// if some `D_k` is not positive or a value is not finite.
fn residuals<T: Float + RealField>(
    data: &SmlData<T>,
    basis: &[LineBasis<T>],
    theta: &[T],
    fixed: usize,
    jacobian: bool,
) -> Option<(DVector<T>, DMatrix<T>)> {
    let p = theta.len();
    let rows = 2 * data.lines.len();
    let mut r = DVector::zeros(rows);
    let mut j = if jacobian { DMatrix::zeros(rows, p) } else { DMatrix::zeros(0, 0) };
    let two = T::one() + T::one();
    for (k, (line, bk)) in data.lines.iter().zip(basis).enumerate() {
        let (a, b) = polynomials(bk, theta, fixed);
        let d = variance(line, a, b);
        if !(d > T::zero() && d.is_finite()) {
            return None;
        }
        let sd = Float::sqrt(d);
        let num = b * line.u - a * line.y;
        let e = num / sd;
        r[2 * k] = e.re;
        r[2 * k + 1] = e.im;
        if jacobian {
            // de = dN / √D - N dD / (2 D √D)
            let mut column = 0;
            let mut put = |dn: Complex<T>, dd: T, column: usize| {
                let de = dn / sd - num * (dd / (two * d * sd));
                j[(2 * k, column)] = de.re;
                j[(2 * k + 1, column)] = de.im;
            };
            for (i, &di) in bk.den.iter().enumerate() {
                if i == fixed {
                    continue;
                }
                let dd = two * line.var_y * (a.conj() * di).re - two * (line.cov_yu * di * b.conj()).re;
                put(-di * line.y, dd, column);
                column += 1;
            }
            for &nj in &bk.num {
                let dd = two * line.var_u * (b.conj() * nj).re - two * (line.cov_yu * a * nj.conj()).re;
                put(nj * line.u, dd, column);
                column += 1;
            }
        }
        if !(e.re.is_finite() && e.im.is_finite()) {
            return None;
        }
    }
    Some((r, j))
}

/// SML cost `Σ |e_k|^2` (infinite where a `D_k` is not positive).
fn cost<T: Float + RealField>(data: &SmlData<T>, basis: &[LineBasis<T>], theta: &[T], fixed: usize) -> T {
    residuals(data, basis, theta, fixed, false).map_or(T::infinity(), |(r, _)| r.dot(&r))
}

/// Columns of `j` scaled to unit norm (in place), and the scales (1 for a zero column).
fn scale_columns<T: Float + RealField>(j: &mut DMatrix<T>) -> Vec<T> {
    (0..j.ncols())
        .map(|c| {
            let norm = j.column(c).norm();
            let sc = if norm > T::zero() { norm } else { T::one() };
            j.column_mut(c).unscale_mut(sc);
            sc
        })
        .collect()
}

/// IQML starting point, then Levenberg-Marquardt (see `identify_continuous`). `fixed` is the
/// index of the coefficient of `A` fixed to 1.
fn fit<T: Float + RealField>(data: &SmlData<T>, basis: &[LineBasis<T>], fixed: usize, options: &SmlOptions<T>) -> Result<Fit<T>, SmlError> {
    let (n, m) = (basis.first().map_or(0, |b| b.den.len() - 1), basis.first().map_or(0, |b| b.num.len() - 1));
    let parameters = n + m + 1;
    if 2 * data.lines.len() <= parameters {
        return Err(SmlError::TooFewLines { lines: data.lines.len(), parameters });
    }
    let rows = 2 * data.lines.len();

    // IQML: weights 1 / D_k of the previous estimate, first of A = 1, B = Ȳ / Ū
    let mut weights: Vec<T> = data
        .lines
        .iter()
        .map(|l| {
            let d = variance(l, Complex::new(T::one(), T::zero()), l.y / l.u);
            T::one() / d
        })
        .collect();
    let mut best: Option<(Vec<T>, T)> = None;
    let mut previous: Option<Vec<T>> = None;
    for _ in 0..options.initial_iterations.max(1) {
        if weights.iter().any(|w| !(*w > T::zero() && w.is_finite())) {
            break;
        }
        let mut lhs = DMatrix::<T>::zeros(rows, parameters);
        let mut rhs = DVector::<T>::zeros(rows);
        for (k, (line, bk)) in data.lines.iter().zip(basis).enumerate() {
            let w = Float::sqrt(weights[k]);
            let mut column = 0;
            for (i, &di) in bk.den.iter().enumerate() {
                let v = di * line.y * w;
                if i == fixed {
                    rhs[2 * k] = v.re;
                    rhs[2 * k + 1] = v.im;
                } else {
                    lhs[(2 * k, column)] = -v.re;
                    lhs[(2 * k + 1, column)] = -v.im;
                    column += 1;
                }
            }
            for &nj in &bk.num {
                let v = nj * line.u * w;
                lhs[(2 * k, column)] = v.re;
                lhs[(2 * k + 1, column)] = v.im;
                column += 1;
            }
        }
        let scale = scale_columns(&mut lhs);
        let Some(Ok(x)) = super::bounded_svd(lhs, true, true).map(|svd| svd.solve(&rhs, T::from(1e-14).unwrap())) else {
            break;
        };
        let theta: Vec<T> = x.iter().zip(&scale).map(|(&v, &sc)| v / sc).collect();
        if theta.iter().any(|v| !v.is_finite()) {
            break;
        }
        let c = cost(data, basis, &theta, fixed);
        if best.as_ref().is_none_or(|(_, best_cost)| c < *best_cost) {
            best = Some((theta.clone(), c));
        }
        let change = previous.as_ref().map(|p: &Vec<T>| {
            let (dd, tt) = p.iter().zip(&theta).fold((T::zero(), T::zero()), |(dd, tt), (&a, &b)| (dd + (a - b) * (a - b), tt + b * b));
            Float::sqrt(dd / tt)
        });
        if change.is_some_and(|c| c <= options.tolerance) {
            break;
        }
        weights = data.lines.iter().zip(basis).map(|(l, bk)| {
            let (a, b) = polynomials(bk, &theta, fixed);
            T::one() / variance(l, a, b)
        }).collect();
        previous = Some(theta);
    }
    let Some((mut theta, mut current)) = best.filter(|(_, c)| c.is_finite()) else {
        return Err(SmlError::Singular);
    };

    // Levenberg-Marquardt with Marquardt scaling: on the unit-norm columns of J, the step
    // δ = -V diag(σ / (σ^2 + μ)) Uᵀ r from the SVD J = U Σ Vᵀ.
    let ten = T::from(10).unwrap();
    let (mu_min, mu_max) = (T::from(1e-12).unwrap(), T::from(1e10).unwrap());
    let mut mu = T::from(1e-3).unwrap();
    let mut iterations = 0;
    let mut termination = Termination::MaxIterations;
    'outer: while iterations < options.max_iterations {
        let Some((r, mut j)) = residuals(data, basis, &theta, fixed, true) else {
            return Err(SmlError::Singular);
        };
        let scale = scale_columns(&mut j);
        let Some(svd) = super::bounded_svd(j, true, true) else {
            return Err(SmlError::Singular);
        };
        let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
        let ur = u.transpose() * &r;
        loop {
            let filtered = DVector::from_iterator(ur.len(), ur.iter().zip(svd.singular_values.iter()).map(|(&x, &s)| x * s / (s * s + mu)));
            let delta_scaled = v_t.transpose() * filtered;
            let delta: Vec<T> = delta_scaled.iter().zip(&scale).map(|(&d, &sc)| -d / sc).collect();
            let trial: Vec<T> = theta.iter().zip(&delta).map(|(&t, &d)| t + d).collect();
            let trial_cost = cost(data, basis, &trial, fixed);
            if trial_cost < current {
                let (dd, tt) = delta.iter().zip(&trial).fold((T::zero(), T::zero()), |(dd, tt), (&d, &t)| (dd + d * d, tt + t * t));
                theta = trial;
                current = trial_cost;
                mu = Float::max(mu / ten, mu_min);
                iterations += 1;
                if dd <= options.tolerance * options.tolerance * tt {
                    termination = Termination::Converged;
                    break 'outer;
                }
                break;
            }
            mu *= ten;
            if mu > mu_max {
                termination = Termination::Stalled;
                break 'outer;
            }
        }
    }

    // Cov(θ) = (M - 2) / (M - 3) · 1/2 (JᵀJ)^-1 (each of Re e, Im e has variance 1/2)
    let Some((_, mut j)) = residuals(data, basis, &theta, fixed, true) else {
        return Err(SmlError::Singular);
    };
    let scale = scale_columns(&mut j);
    let Some(svd) = super::bounded_svd(j, false, true) else {
        return Err(SmlError::Singular);
    };
    let v_t = svd.v_t.unwrap();
    let mm = T::from(data.periods).unwrap();
    let factor = (mm - T::from(2).unwrap()) / (mm - T::from(3).unwrap()) / T::from(2).unwrap();
    let std: Vec<T> = (0..parameters)
        .map(|c| {
            let var = svd.singular_values.iter().enumerate().fold(T::zero(), |acc, (i, &s)| acc + v_t[(i, c)] * v_t[(i, c)] / (s * s));
            Float::sqrt(factor * var) / scale[c]
        })
        .collect();

    Ok(Fit { theta, std, cost: current, parameters, iterations, termination })
}
