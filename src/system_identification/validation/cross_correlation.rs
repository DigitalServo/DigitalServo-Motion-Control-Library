//! Cross-correlation test of the residual and the input.

use num_complex::Complex;
use num_traits::Float;
use rustfft::{FftNum, FftPlanner};

use super::Validation;

impl<T: Float + FftNum> Validation<T> {
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
    /// If the residual or the input is identically zero after the mean is removed, `r(τ)` is zero
    /// and the bound is that of a white residual, `z / sqrt(N)`.
    pub fn cross_correlation(&self, max_lag: usize, z: T) -> CrossCorrelation<T> {
        let (start, end) = (self.start.min(self.residual.len()), self.residual.len());
        let n = T::from(end - start).unwrap();
        let mean = |x: &[T]| x.iter().fold(T::zero(), |acc, &v| acc + v) / n;
        let (e_mean, u_mean) = (mean(&self.residual[start..]), mean(&self.input[start..]));
        let e: Vec<T> = self.residual.iter().map(|&v| v - e_mean).collect();
        let u: Vec<T> = self.input.iter().map(|&v| v - u_mean).collect();

        // R_εu(τ) = Σ_{k in evaluated range, 0 <= k - τ < len} ε[k] u[k - τ] / N: the input before
        // `start` is known (past of the evaluated samples); the residual is not used there
        let e_evaluated: Vec<T> = (0..end).map(|k| if k >= start { e[k] } else { T::zero() }).collect();
        let r_eu: Vec<T> = correlation(&e_evaluated, &u, max_lag).iter().map(|&c| c / n).collect();
        // Autocorrelations over the evaluated samples only, at the lags 0 ..= max_lag
        let autocorrelation = |x: &[T]| -> Vec<T> { correlation(x, x, max_lag)[max_lag..].iter().map(|&c| c / n).collect() };
        let (r_e, r_u) = (autocorrelation(&e[start..]), autocorrelation(&u[start..]));
        let max_lag = max_lag as isize;
        let lags: Vec<isize> = (-max_lag..=max_lag).collect();

        // Normalized signal by signal (ρ_ε(k) = R_ε(k) / R_ε(0), P / (R_ε(0) R_u(0)) = Σ_k ρ_ε(k) ρ_u(k)),
        // so that the products of small powers do not underflow
        let (scale_e, scale_u) = (r_e[0].sqrt(), r_u[0].sqrt());
        if !(scale_e > T::zero() && scale_u > T::zero() && n > T::zero()) {
            // A residual or an input that is identically zero (or constant): no correlation to
            // test, zero rather than 0 / 0, with the bound of a white residual
            return CrossCorrelation { correlation: vec![T::zero(); lags.len()], lags, bound: z / n.max(T::one()).sqrt() };
        }
        let (rho_e, rho_u) = (r_e.iter().map(|&r| r / r_e[0]), r_u.iter().map(|&r| r / r_u[0]));
        let p = rho_e.zip(rho_u).skip(1).enumerate().fold(T::one(), |acc, (i, (re, ru))| {
            let w = T::one() - T::from(i + 1).unwrap() / T::from(max_lag + 1).unwrap();
            acc + (T::one() + T::one()) * w * re * ru
        });
        CrossCorrelation {
            lags,
            correlation: r_eu.iter().map(|&r| r / scale_e / scale_u).collect(),
            bound: z * (p.max(T::zero()) / n).sqrt(),
        }
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

/// Linear correlation `c[τ] = Σ_k a[k] b[k - τ]` (zero outside the sequences) at the lags
/// `τ = -max_lag ..= max_lag`, by FFT: `c = IFFT(A conj(B)) / L_fft` with `A`, `B` the FFTs of the
/// zero-padded sequences, `L_fft >= len + max_lag` (a power of two) so that the circular
/// correlation does not wrap around within the lags.
fn correlation<T: Float + FftNum>(a: &[T], b: &[T], max_lag: usize) -> Vec<T> {
    let len = a.len().max(b.len());
    let size = (len + max_lag).next_power_of_two();
    let mut planner = FftPlanner::new();
    let (forward, inverse) = (planner.plan_fft_forward(size), planner.plan_fft_inverse(size));
    let padded = |x: &[T]| {
        let mut buffer = vec![Complex::new(T::zero(), T::zero()); size];
        for (b, &v) in buffer.iter_mut().zip(x) {
            b.re = v;
        }
        buffer
    };
    let (mut fa, mut fb) = (padded(a), padded(b));
    forward.process(&mut fa);
    forward.process(&mut fb);
    let mut product: Vec<Complex<T>> = fa.iter().zip(&fb).map(|(x, y)| x * y.conj()).collect();
    inverse.process(&mut product);
    let scale = T::from(size).unwrap();
    // Lag τ at index τ mod size
    (0..=2 * max_lag).map(|i| product[(i + size - max_lag) % size].re / scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_by_fft_matches_the_direct_sum() {
        let a: Vec<f64> = (0..300).map(|k| ((k * 7919) % 97) as f64 / 97.0 - 0.5).collect();
        let b: Vec<f64> = (0..300).map(|k| ((k * 104729) % 89) as f64 / 89.0 - 0.5).collect();
        let max_lag = 40;
        let c = correlation(&a, &b, max_lag);
        for (i, tau) in (-(max_lag as isize)..=max_lag as isize).enumerate() {
            let direct: f64 = (0..a.len() as isize)
                .filter(|&k| k - tau >= 0 && k - tau < b.len() as isize)
                .map(|k| a[k as usize] * b[(k - tau) as usize])
                .sum();
            assert!((c[i] - direct).abs() < 1e-10, "τ = {tau}: {} vs {direct}", c[i]);
        }
    }
}
