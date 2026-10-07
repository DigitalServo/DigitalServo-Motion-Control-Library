//! Test of the residual at the excited lines of a periodic input.

use num_complex::Complex;
use num_traits::Float;
use rustfft::{FftNum, FftPlanner};

use super::{Validation, ValidationError};

impl<T: Float + FftNum> Validation<T> {
    /// Test of the residual at the excited lines of a periodic input (e.g. a multisine) of
    /// fundamental frequency `fundamental_frequency` \[Hz\] (period `1 / f0`, a whole number of
    /// samples), over the whole periods from `start` on (set it by `evaluated_from` after the
    /// transient of the plant from rest, e.g. one period, if the data are also used for the noise
    /// level). The lines are the harmonic numbers of the fundamental (frequencies `line f0` \[Hz\]),
    /// as for `multisine`.
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
    pub fn line_test(&self, fundamental_frequency: T, lines: &[usize], confidence: T) -> Result<LineTest<T>, ValidationError> {
        let period = self.period_samples(fundamental_frequency)?;
        let start = self.start.min(self.residual.len());
        let periods = (self.residual.len() - start) / period.max(1);
        if periods < 2 {
            return Err(ValidationError::TooFewPeriods { periods });
        }
        if let Some(&line) = lines.iter().find(|&&l| l == 0 || 2 * l > period) {
            return Err(ValidationError::InvalidLine { line, half: period / 2 });
        }

        let p = T::from(periods).unwrap();
        let measured: Vec<T> = self.simulated.iter().zip(&self.residual).map(|(&a, &b)| a + b).collect();

        // DFT of every period by one FFT each (`E_p(f) = Σ_i x[i] e^(-j 2π f i / period)`, the
        // forward FFT without scaling), the lines picked from the bins
        let fft = FftPlanner::new().plan_fft_forward(period);
        let spectra = |x: &[T]| -> Vec<Vec<Complex<T>>> {
            (0..periods)
                .map(|index| {
                    let offset = start + index * period;
                    let mut buffer: Vec<Complex<T>> = x[offset..offset + period].iter().map(|&v| Complex::new(v, T::zero())).collect();
                    fft.process(&mut buffer);
                    lines.iter().map(|&line| buffer[line]).collect()
                })
                .collect()
        };
        let (residual_lines, output_lines) = (spectra(&self.residual), spectra(&measured));

        let mut test = LineTest {
            lines: lines.to_vec(),
            fundamental_frequency,
            periods,
            residual: Vec::with_capacity(lines.len()),
            output: Vec::with_capacity(lines.len()),
            noise_variance: Vec::with_capacity(lines.len()),
            statistic: Vec::with_capacity(lines.len()),
            bound: (p - T::one()) * ((T::one() - confidence).powf(-T::one() / (p - T::one())) - T::one()),
        };
        let zero = Complex::new(T::zero(), T::zero());
        for l in 0..lines.len() {
            let mean = residual_lines.iter().fold(zero, |acc, e| acc + e[l]) / p;
            let variance = residual_lines.iter().fold(T::zero(), |acc, e| acc + (e[l] - mean).norm_sqr()) / (p - T::one());
            let output = output_lines.iter().fold(zero, |acc, y| acc + y[l]) / p;
            // Without noise (variance zero) a model error is infinitely significant, and no
            // error at all is zero rather than 0 / 0
            let statistic = if variance > T::zero() {
                p * mean.norm_sqr() / variance
            } else if mean.norm_sqr() > T::zero() {
                T::infinity()
            } else {
                T::zero()
            };
            test.statistic.push(statistic);
            test.residual.push(mean);
            test.output.push(output);
            test.noise_variance.push(variance);
        }
        Ok(test)
    }
}

/// Result of `Validation::line_test`.
#[derive(Clone, Debug)]
pub struct LineTest<T> {
    /// Excited lines (harmonic numbers of the period).
    pub lines: Vec<usize>,
    /// Fundamental frequency \[Hz\].
    pub fundamental_frequency: T,
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
    /// Frequencies \[Hz\] of the lines, `line f0`.
    pub fn frequencies(&self) -> Vec<T> {
        self.lines.iter().map(|&l| T::from(l).unwrap() * self.fundamental_frequency).collect()
    }

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
