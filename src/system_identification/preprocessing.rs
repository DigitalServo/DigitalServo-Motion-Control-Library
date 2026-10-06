//! Preprocessing of identification data.

use num_traits::Float;

/// High-pass filtering of the input `u` and the output `y` alike, to remove a drift (e.g. the ramp
/// of an integrating plant driven by an unknown input offset) before identification.
///
/// `order` first-order high-pass filters with cutoff `cutoff` \[Hz\] (backward Euler,
/// `x_f[k] = a (x_f[k-1] + x[k] - x[k-1])`, `a = 1 / (1 + 2π cutoff ts)`) are applied in series to
/// both signals, starting at rest (`x[-1] = x_f[-1] = 0`). Put the cutoff below the excited band.
///
/// - The same filter on both: the plant is linear, so the filtered signals are related by the same
///   `G(s)`; a filtered sampled input is still constant between samples, so the zero-order-hold
///   assumption of the identification (e.g. SRIVC) stays exact.
/// - Starting at rest: the step of the signals at `k = 0` is filtered too. Starting from
///   `x[-1] = x[0]` instead breaks the relation between the filtered input and output (errors of
///   5 .. 20 % on a two-inertia plant).
pub fn high_pass<T: Float>(u: &[T], y: &[T], cutoff: T, ts: T, order: usize) -> (Vec<T>, Vec<T>) {
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
    let a = T::one() / (T::one() + two_pi * cutoff * ts);
    let filter = |x: &[T]| {
        let mut x = x.to_vec();
        for _ in 0..order {
            let (mut previous_in, mut previous_out) = (T::zero(), T::zero());
            for v in x.iter_mut() {
                let out = a * (previous_out + *v - previous_in);
                previous_in = *v;
                previous_out = out;
                *v = out;
            }
        }
        x
    };
    (filter(u), filter(y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_a_constant_and_keeps_high_frequencies() {
        let ts = 1e-3;
        // A constant offset decays; a 50 Hz sine passes (cutoff 1 Hz)
        let x: Vec<f64> = (0..5000).map(|k| 1.0 + (2.0 * std::f64::consts::PI * 50.0 * k as f64 * ts).sin()).collect();
        let (xf, _) = high_pass(&x, &x, 1.0, ts, 2);
        let tail = &xf[4000..];
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        let amplitude = tail.iter().fold(0.0f64, |m, v| m.max((v - mean).abs()));
        assert!(mean.abs() < 1e-3 && (amplitude - 1.0).abs() < 0.01, "mean {mean}, amplitude {amplitude}");
    }
}
