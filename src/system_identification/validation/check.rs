//! Running several criteria and tests at once with one confidence level (`Validation::check`).

use std::fmt;

use num_traits::Float;
use rustfft::FftNum;

use super::{CoherenceTest, CrossCorrelation, LineTest, Validation, ValidationError, WhitenessTest};

/// A criterion or test to run by `Validation::check`.
#[derive(Clone, Debug, PartialEq)]
pub enum Check<T> {
    /// `mse`, `bic`, `aic`, `aicc` with the number of estimated parameters (no pass / fail).
    InformationCriteria { parameters: usize },
    /// Whiteness of the residual over the lags `1 ..= max_lag` (`autocorrelation`), by the
    /// Ljung-Box statistic.
    Whiteness { max_lag: usize },
    /// Independence of the residual and the input over the lags `-max_lag ..= max_lag`
    /// (`cross_correlation`).
    CrossCorrelation { max_lag: usize },
    /// Coherence of the residual and the input (`coherence_test`) over the bins where the input
    /// power is at least `excited` times its maximum (e.g. `1e-2`; 0 for all bins).
    Coherence { segment_len: usize, excited: T },
    /// Test at the excited `lines` of a periodic input of `period` samples (`line_test`), over the
    /// whole periods from `Validation::start` on (set it after the transient, e.g. one period).
    Lines { period: usize, lines: Vec<usize> },
}

/// Result of one `Check`.
#[derive(Clone, Debug)]
pub enum CheckResult<T> {
    /// Information criteria of the evaluated residual.
    InformationCriteria { samples: usize, parameters: usize, mse: T, bic: T, aic: T, aicc: T },
    /// Whiteness: passed if the Ljung-Box statistic is at most `bound`, the `confidence` quantile
    /// of `χ²(max_lag)`.
    Whiteness { test: WhitenessTest<T>, bound: T, passed: bool },
    /// Cross-correlation: passed if no lag is outside the bound, which is at the Bonferroni level
    /// `(1 - confidence) / M` per lag (`M` lags).
    CrossCorrelation { test: CrossCorrelation<T>, passed: bool },
    /// Coherence: passed if at most `allowed` bins are outside the per-bin bound.
    Coherence { test: CoherenceTest<T>, allowed: usize, passed: bool },
    /// Line test: passed if at most `allowed` lines are outside the per-line bound.
    Lines { test: LineTest<T>, allowed: usize, passed: bool },
}

impl<T> CheckResult<T> {
    /// Pass / fail (`None` for the information criteria).
    pub fn passed(&self) -> Option<bool> {
        match self {
            Self::InformationCriteria { .. } => None,
            Self::Whiteness { passed, .. }
            | Self::CrossCorrelation { passed, .. }
            | Self::Coherence { passed, .. }
            | Self::Lines { passed, .. } => Some(*passed),
        }
    }
}

/// Results of `Validation::check`, in the order of the checks.
#[derive(Clone, Debug)]
pub struct Report<T> {
    /// Confidence level of the tests.
    pub confidence: T,
    /// One result per check.
    pub results: Vec<CheckResult<T>>,
}

impl<T> Report<T> {
    /// Whether every test passed (the information criteria have no pass / fail).
    pub fn passed(&self) -> bool {
        self.results.iter().all(|r| r.passed() != Some(false))
    }
}

impl<T: Float + FftNum> Validation<T> {
    /// Run `checks` with one `confidence` level (e.g. `0.99`), with the pass / fail rules:
    ///
    /// - whiteness: the Ljung-Box statistic within the `confidence` quantile of `χ²(max_lag)`
    ///   (one test over all lags);
    /// - cross-correlation: `r(τ)` at neighboring lags is strongly correlated when the input is
    ///   colored (for a white residual, as the input autocorrelation `ρ_u(τ1 - τ2)`), so lags
    ///   outside come in clusters and their number is not binomial (that rule rejected a right
    ///   model 12 % of the time at 99 % with a band-limited input). Every lag is therefore tested
    ///   at the Bonferroni level `(1 - confidence) / M` (`M = 2 max_lag + 1` lags, two-sided), and
    ///   the check passes if none is outside: the false rejection rate is at most
    ///   `1 - confidence` whatever the correlation;
    /// - coherence, lines: each bin / line is tested at `confidence`, and a right model still has
    ///   about `1 - confidence` of them outside by chance. The check passes if the number outside
    ///   is at most the `confidence` quantile of the binomial distribution `B(M, 1 - confidence)`
    ///   (`M` bins / lines; the lines are independent, neighboring bins nearly so).
    ///
    /// The individual results (`CheckResult`) keep the full tests for a closer look, e.g. at which
    /// lags or frequencies a test failed.
    pub fn check(&self, checks: &[Check<T>], confidence: T) -> Result<Report<T>, ValidationError> {
        let alpha = T::one() - confidence;
        let z_two_sided = normal_quantile(T::one() - alpha / T::from(2).unwrap());
        let z_one_sided = normal_quantile(confidence);
        let results = checks
            .iter()
            .map(|check| {
                Ok(match check {
                    Check::InformationCriteria { parameters } => CheckResult::InformationCriteria {
                        samples: self.samples(),
                        parameters: *parameters,
                        mse: self.mse(),
                        bic: self.bic(*parameters),
                        aic: self.aic(*parameters),
                        aicc: self.aicc(*parameters),
                    },
                    Check::Whiteness { max_lag } => {
                        let test = self.autocorrelation(*max_lag, z_two_sided);
                        let bound = test.ljung_box_bound(z_one_sided);
                        let passed = test.ljung_box <= bound;
                        CheckResult::Whiteness { test, bound, passed }
                    }
                    Check::CrossCorrelation { max_lag } => {
                        let lags = T::from(2 * max_lag + 1).unwrap();
                        let z = normal_quantile(T::one() - alpha / (T::from(2).unwrap() * lags));
                        let test = self.cross_correlation(*max_lag, z);
                        let passed = test.outside().is_empty();
                        CheckResult::CrossCorrelation { test, passed }
                    }
                    Check::Coherence { segment_len, excited } => {
                        let test = self.coherence_test(*segment_len, confidence)?.excited(*excited);
                        let allowed = binomial_quantile(test.bins.len(), alpha, confidence);
                        let passed = test.outside().len() <= allowed;
                        CheckResult::Coherence { test, allowed, passed }
                    }
                    Check::Lines { period, lines } => {
                        let test = self.line_test(*period, lines, confidence)?;
                        let allowed = binomial_quantile(test.lines.len(), alpha, confidence);
                        let passed = test.outside().len() <= allowed;
                        CheckResult::Lines { test, allowed, passed }
                    }
                })
            })
            .collect::<Result<Vec<_>, ValidationError>>()?;
        Ok(Report { confidence, results })
    }
}

impl<T: Float + fmt::Display + fmt::LowerExp> fmt::Display for Report<T> {
    /// One line per check, then the overall verdict.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verdict = |passed: bool| if passed { "passed" } else { "FAILED" };
        let hundred = T::from(100).unwrap();
        writeln!(f, "validation at {:.1} % confidence", self.confidence * hundred)?;
        for result in &self.results {
            match result {
                CheckResult::InformationCriteria { samples, parameters, mse, bic, aic, aicc } => writeln!(
                    f,
                    "  information criteria: N = {samples}, p = {parameters}, V = {mse:.4e}, BIC = {bic:.1}, AIC = {aic:.1}, AICc = {aicc:.1}"
                )?,
                CheckResult::Whiteness { test, bound, passed } => writeln!(
                    f,
                    "  whiteness ({} lags): Ljung-Box Q = {:.1} (bound {bound:.1}), {} lags outside: {}",
                    test.lags.len(),
                    test.ljung_box,
                    test.outside().len(),
                    verdict(*passed)
                )?,
                CheckResult::CrossCorrelation { test, passed } => writeln!(
                    f,
                    "  cross-correlation ({} lags, Bonferroni bound {:.4}): {} outside, max |r| / bound {:.2}: {}",
                    test.lags.len(),
                    test.bound,
                    test.outside().len(),
                    test.max_ratio(),
                    verdict(*passed)
                )?,
                CheckResult::Coherence { test, allowed, passed } => writeln!(
                    f,
                    "  coherence ({} bins): {} outside (allowed {allowed}), mean γ² {:.4} (expected {:.4}): {}",
                    test.bins.len(),
                    test.outside().len(),
                    test.mean_coherence(),
                    test.expected_coherence(),
                    verdict(*passed)
                )?,
                CheckResult::Lines { test, allowed, passed } => writeln!(
                    f,
                    "  lines ({} lines, {} periods): {} outside (allowed {allowed}), mean F {:.2} (expected {:.2}): {}",
                    test.lines.len(),
                    test.periods,
                    test.outside().len(),
                    test.mean_statistic(),
                    test.expected_statistic(),
                    verdict(*passed)
                )?,
            }
        }
        write!(f, "  overall: {}", verdict(self.passed()))
    }
}

/// Quantile `Φ^-1(p)` of the standard normal distribution (Acklam's rational approximation,
/// relative error below 1.2e-9), for `0 < p < 1`.
fn normal_quantile<T: Float>(p: T) -> T {
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let p = p.to_f64().unwrap();
    let tail = |q: f64| {
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    };
    let x = if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num / den
    };
    T::from(x).unwrap()
}

/// Smallest `k` with `P(X <= k) >= confidence` for `X ~ B(m, alpha)`.
fn binomial_quantile<T: Float>(m: usize, alpha: T, confidence: T) -> usize {
    let (alpha, confidence) = (alpha.to_f64().unwrap(), confidence.to_f64().unwrap());
    let mut pmf = (1.0 - alpha).powi(m as i32);
    let mut cdf = pmf;
    let mut k = 0;
    while cdf < confidence && k < m {
        // P(k + 1) = P(k) (m - k) / (k + 1) α / (1 - α)
        pmf *= (m - k) as f64 / (k + 1) as f64 * alpha / (1.0 - alpha);
        k += 1;
        cdf += pmf;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_quantile_values() {
        for (p, z) in [(0.5, 0.0), (0.975, 1.959964), (0.995, 2.575829), (0.99, 2.326348), (0.01, -2.326348), (1e-6, -4.753424)] {
            assert!((normal_quantile(p) - z).abs() < 1e-5, "Φ^-1({p}) = {} vs {z}", normal_quantile(p));
        }
    }

    #[test]
    fn binomial_quantile_values() {
        // B(100, 0.01): P(X <= 2) = 0.9206, P(X <= 3) = 0.9816, P(X <= 4) = 0.9966
        assert_eq!(binomial_quantile(100, 0.01, 0.95), 3);
        assert_eq!(binomial_quantile(100, 0.01, 0.99), 4);
        assert_eq!(binomial_quantile(10, 0.05, 0.5), 0);
    }
}
