//! Whiteness test of the residual (autocorrelation, Ljung-Box).

use num_traits::Float;

use super::Validation;

impl<T: Float> Validation<T> {
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
