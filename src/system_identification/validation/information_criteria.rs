//! Information criteria: BIC, AIC, AICc.

use num_traits::Float;

use super::Validation;

impl<T: Float> Validation<T> {
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
}
