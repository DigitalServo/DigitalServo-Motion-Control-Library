//! Excitation signals for identification experiments.

use num_traits::Float;

/// Multisine of `n_samples` samples, periodic with `period` samples: the sum of the `harmonics` of
/// the period (frequencies `h / (period ts)`) with random phases (reproducible from `seed`) and
/// equal amplitudes, scaled to unit RMS:
///
/// ```text
/// u[k] = sqrt(2 / H) Σ_h sin(2π h k / period + φ_h),   φ_h uniform in [0, 2π)
/// ```
///
/// With whole periods, all the power is at the excited lines (no leakage), and the period-to-period
/// variation of the response gives the noise level at every line (`Validation::line_test`). The
/// first period can be left out of an evaluation as the transient from rest.
pub fn multisine<T: Float>(n_samples: usize, period: usize, harmonics: &[usize], seed: u64) -> Vec<T> {
    let mut state = seed;
    let phases: Vec<f64> = harmonics.iter().map(|_| 2.0 * std::f64::consts::PI * uniform(&mut state)).collect();
    let scale = (2.0 / harmonics.len().max(1) as f64).sqrt();
    (0..n_samples)
        .map(|k| {
            let value: f64 = harmonics
                .iter()
                .zip(&phases)
                .map(|(&h, &phase)| (2.0 * std::f64::consts::PI * ((h * k) % period) as f64 / period as f64 + phase).sin())
                .sum();
            T::from(scale * value).unwrap()
        })
        .collect()
}

/// Linear chirp (swept sine) of unit amplitude from `f0` to `f1` \[Hz\] over `n_samples` samples
/// of period `ts`:
///
/// ```text
/// u[k] = sin(2π (f0 t + (f1 - f0) t^2 / (2 T))),   t = k ts,  T = n_samples ts
/// ```
///
/// Not periodic: validate with `Validation::coherence_test` / `cross_correlation`, not `line_test`.
pub fn chirp<T: Float>(n_samples: usize, ts: T, f0: T, f1: T) -> Vec<T> {
    let (ts, f0, f1) = (ts.to_f64().unwrap(), f0.to_f64().unwrap(), f1.to_f64().unwrap());
    let duration = n_samples as f64 * ts;
    (0..n_samples)
        .map(|k| {
            let t = k as f64 * ts;
            T::from((2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) * t * t / (2.0 * duration))).sin()).unwrap()
        })
        .collect()
}

/// splitmix64: uniform in [0, 1).
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multisine_is_periodic_with_unit_rms() {
        let (period, harmonics) = (1000, (10..=60).collect::<Vec<usize>>());
        let u: Vec<f64> = multisine(3 * period, period, &harmonics, 7);
        assert!((0..period).all(|k| (u[k] - u[k + period]).abs() < 1e-12 && (u[k] - u[k + 2 * period]).abs() < 1e-12));
        let rms = (u[..period].iter().map(|v| v * v).sum::<f64>() / period as f64).sqrt();
        assert!((rms - 1.0).abs() < 1e-12, "rms {rms}");
        // Reproducible from the seed, different for another seed
        assert_eq!(u, multisine::<f64>(3 * period, period, &harmonics, 7));
        assert_ne!(u, multisine::<f64>(3 * period, period, &harmonics, 8));
    }

    #[test]
    fn chirp_sweeps_linearly() {
        let u: Vec<f64> = chirp(1000, 1e-3, 1.0, 11.0);
        assert!(u[0].abs() < 1e-12 && u.iter().all(|v| v.abs() <= 1.0));
        // Instantaneous frequency f0 + (f1 - f0) t / T: 6 Hz in the middle, i.e. zero crossings
        // about 1 / 12 s apart there
        let crossings: Vec<usize> = (450..560).filter(|&k| u[k] <= 0.0 && u[k + 1] > 0.0 || u[k] >= 0.0 && u[k + 1] < 0.0).collect();
        let spacing = (crossings[crossings.len() - 1] - crossings[0]) as f64 / (crossings.len() - 1) as f64 * 1e-3;
        assert!((spacing - 1.0 / 12.0).abs() < 0.01, "spacing {spacing}");
    }
}
