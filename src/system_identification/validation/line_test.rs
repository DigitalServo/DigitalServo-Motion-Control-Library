//! Test of the residual at the excited lines of a periodic input.

use num_complex::Complex;
use num_traits::Float;

use super::{Validation, ValidationError};

impl<T: Float> Validation<T> {
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
