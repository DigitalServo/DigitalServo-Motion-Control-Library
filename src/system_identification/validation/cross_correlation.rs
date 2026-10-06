//! Cross-correlation test of the residual and the input.

use num_traits::Float;

use super::Validation;

impl<T: Float> Validation<T> {
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
