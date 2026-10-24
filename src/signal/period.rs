//! Period of a periodic signal from its autocorrelation.

use num_traits::Float;
use rustfft::{num_complex::Complex, FftNum, FftPlanner};

/// Autocorrelation below which the main lobe (around lag 0) ends, and below which the highest
/// peak after it means that the signal is not periodic.
pub const PERIOD_CORRELATION_THRESHOLD: f64 = 0.5;

/// Fraction of the highest autocorrelation peak a peak must reach to be taken as the period: the
/// first such peak is the shortest period, not a multiple of it.
pub const PERIOD_PEAK_FRACTION: f64 = 0.9;

/// Period \[samples\] of `x` (e.g. the input of a periodic excitation), at most `max_period`, or
/// `None` if `x` is not periodic.
///
/// From the unbiased normalized autocorrelation of `x` with its mean removed,
///
/// ```text
/// r(τ) = (Σ_k x[k] x[k + τ] / (n - τ)) / (Σ_k x[k]^2 / n)
/// ```
///
/// (by FFT of `x` zero-padded to `2n`, `O(n log n)`): the main lobe ends at the first lag where
/// `r < PERIOD_CORRELATION_THRESHOLD` (0.5). If the largest `r` after it is below the same 0.5,
/// `x` is not periodic. Otherwise the period is the shortest lag of a local maximum with
/// `r(τ) >= PERIOD_PEAK_FRACTION · max` (0.9), i.e. the shortest period, not a multiple of it.
/// The lags go up to `min(max_period, n - 1)`, and a local maximum at the last of them must not be
/// followed by a higher `r` (at lag `max_period + 1`, if `max_period < n - 1`): `None` if no peak
/// qualifies, e.g. when the period is longer than `max_period` and `r` is still rising there. A
/// transient at the start or noise lowers the peaks but not their lags.
///
/// ```
/// use dsmc::signal::detect_period;
///
/// let x: Vec<f64> = (0..2000).map(|k| (2.0 * std::f64::consts::PI * k as f64 / 50.0).sin() + (k % 50) as f64 * 0.01).collect();
/// assert_eq!(detect_period(&x, 500), Some(50));
/// let ramp: Vec<f64> = (0..2000).map(|k| k as f64).collect();
/// assert_eq!(detect_period(&ramp, 500), None);
/// ```
pub fn detect_period<T: Float + FftNum>(x: &[T], max_period: usize) -> Option<usize> {
    let n = x.len();
    let last = max_period.min(n.checked_sub(1)?);
    if last < 2 {
        return None;
    }
    let mean = x.iter().fold(T::zero(), |acc, &v| acc + v) / T::from(n).unwrap();
    let mut buffer: Vec<Complex<T>> = x.iter().map(|&v| Complex::new(v - mean, T::zero())).collect();
    buffer.resize(2 * n, Complex::new(T::zero(), T::zero()));
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(2 * n).process(&mut buffer);
    for v in buffer.iter_mut() {
        *v = Complex::new(v.norm_sqr(), T::zero());
    }
    planner.plan_fft_inverse(2 * n).process(&mut buffer);
    // The unnormalized inverse FFT: buffer[τ] = 2n Σ_k x[k] x[k + τ]
    let energy = buffer[0].re;
    if !(energy > T::zero() && energy.is_finite()) {
        return None;
    }
    let len = T::from(n).unwrap();
    let r: Vec<T> = (0..=(last + 1).min(n - 1)).map(|tau| buffer[tau].re / energy * len / T::from(n - tau).unwrap()).collect();

    let threshold = T::from(PERIOD_CORRELATION_THRESHOLD).unwrap();
    let start = (1..=last).find(|&tau| r[tau] < threshold)?;
    let max = (start..=last).fold(r[start], |m, tau| Float::max(m, r[tau]));
    if max < threshold {
        return None;
    }
    let level = T::from(PERIOD_PEAK_FRACTION).unwrap() * max;
    let local_max = |tau: usize| r[tau] >= r[tau - 1] && r.get(tau + 1).is_none_or(|&next| r[tau] >= next);
    (start..=last).find(|&tau| r[tau] >= level && local_max(tau))
}
