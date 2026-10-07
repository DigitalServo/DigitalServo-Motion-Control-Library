//! Excitation signals for identification experiments.

use num_traits::{Float, FloatConst};
use thiserror::Error;

use crate::FrequencyTransferFunction;
use crate::sampling::{samples_per_period, whole_samples, DurationError};

/// Multisine of `tlen` \[s\] sampled with period `ts` \[s\]: the sum of the `harmonics` `h` of the
/// fundamental frequency `fundamental_frequency` \[Hz\] (frequencies `h f0`), whose amplitudes are
/// proportional to `amplitude(h)`, with random phases (reproducible from `seed`), scaled to unit RMS:
///
/// ```text
/// u[k] = c Σ_h a_h sin(θ_h (k + 1/2) + ψ_h),   θ_h = 2π h / N,   N = 1 / (f0 ts),   a_h = amplitude(h)
/// c    = 1 / sqrt(Σ_h a_h^2 / 2)
/// ψ_h  = φ_h + θ_h τ,   φ_h uniform in [0, 2π),   τ ∈ [0, N): the first root of
///        D(τ) = Σ_h a_h cos(φ_h + θ_h τ) / (2 sin(θ_h / 2)) = 0
/// ```
///
/// The signal is periodic with `N` samples: with whole periods, all the power is at the excited
/// lines (no leakage), and the period-to-period variation of the response gives the noise level at
/// every line (`Validation::line_test`). The first period can be left out of an evaluation as the
/// transient from rest.
///
/// Equal amplitudes (`|_| 1.0`) spread the power evenly over the band; shaping them, e.g. like
/// `1 / |G|` (see `flat_output_multisine`), puts the input power where the plant needs it, so that
/// every line reaches a similar signal-to-noise ratio at the output.
///
/// The time shift `τ` of the random-phase signal keeps a plant with up to two integrators, started
/// from rest, from drifting: the sum of the input `Σ_{k<n} u[k] = c (D(τ) - Σ_h a_h cos(θ_h n + ψ_h)
/// / (2 sin(θ_h / 2)))` (by `Σ_{k<n} sin(θ (k + 1/2) + ψ) = (cos ψ - cos(θ n + ψ)) / (2 sin(θ / 2))`)
/// has no constant term when `D(τ) = 0`, so the velocity of a double integrator does not have one
/// either and its position no ramp (only a constant offset), for a discrete integrator
/// (accumulator) and for a continuous one behind a zero-order hold alike. `D` has zero mean over a
/// period, so it has a root.
///
/// The period `N` and `tlen / ts` must be whole numbers of samples, i.e. the sampling frequency a
/// whole multiple of the fundamental (e.g. not 3 Hz at 1 kHz: 333.3 samples; relative tolerance
/// `max(1e-9, 4 eps)`). The harmonics must be distinct, at least one, and strictly below the Nyquist
/// frequency (`1 <= h < N / 2`), and `amplitude(h)` finite and not zero for all of them. The unit RMS
/// holds over whole periods.
///
/// ```
/// use dsmc::signal::excitation::multisine;
///
/// // 1 ..= 40 Hz with equal amplitudes, fundamental 1 Hz at 1 kHz, 3 periods
/// let lines: Vec<usize> = (1..=40).collect();
/// let u: Vec<f64> = multisine(3.0, 1e-3, 1.0, &lines, |_| 1.0, 1).unwrap();
/// assert_eq!(u.len(), 3000);
/// assert!((0..2000).all(|k| (u[k] - u[k + 1000]).abs() < 1e-12));
/// let rms = (u[..1000].iter().map(|v| v * v).sum::<f64>() / 1000.0).sqrt();
/// assert!((rms - 1.0).abs() < 1e-12);
///
/// // Amplitudes rising like the frequency (e.g. 1 / |G| of an integrator)
/// let u: Vec<f64> = multisine(3.0, 1e-3, 1.0, &lines, |h| h as f64, 1).unwrap();
/// ```
pub fn multisine<T: Float>(
    tlen: T,
    ts: T,
    fundamental_frequency: T,
    harmonics: &[usize],
    amplitude: impl Fn(usize) -> T,
    seed: u64,
) -> Result<Vec<T>, ExcitationError> {
    let n_samples = whole_samples(tlen, ts).map_err(|kind| duration_error(kind, tlen, ts))?;
    let period = samples_per_period(fundamental_frequency, ts).map_err(|kind| {
        let (fundamental_frequency, ts) = (fundamental_frequency.to_f64().unwrap_or(f64::NAN), ts.to_f64().unwrap_or(f64::NAN));
        match kind {
            DurationError::Invalid => ExcitationError::InvalidFundamental { fundamental_frequency, ts },
            DurationError::Fractional => ExcitationError::FractionalPeriod { fundamental_frequency, ts },
        }
    })?;
    check_harmonics(period, harmonics)?;
    let mut state = seed;
    let mut lines: Vec<(usize, f64, f64)> = Vec::with_capacity(harmonics.len());
    for &h in harmonics {
        let a = amplitude(h).to_f64().filter(|a| a.is_finite() && *a != 0.0).ok_or(ExcitationError::InvalidAmplitude { harmonic: h })?;
        lines.push((h, a, 2.0 * std::f64::consts::PI * uniform(&mut state)));
    }
    // RMS of a sum of sinusoids of distinct frequencies (0 < h < N / 2): sqrt(Σ a^2 / 2)
    let power = lines.iter().map(|(_, a, _)| a * a).sum::<f64>() / 2.0;
    if !(power > 0.0 && power.is_finite()) {
        return Err(ExcitationError::NoPower);
    }
    let scale = 1.0 / power.sqrt();

    // Time shift τ with D(τ) = 0: D has zero mean over the integers 0 .. N - 1 too, so it changes
    // sign between two of them (or vanishes at one); the root between them by bisection
    let theta = |h: usize| 2.0 * std::f64::consts::PI * h as f64 / period as f64;
    let drift = |tau: f64| lines.iter().map(|&(h, a, phi)| a * (phi + theta(h) * tau).cos() / (2.0 * (theta(h) / 2.0).sin())).sum::<f64>();
    let bisect = |mut lo: f64, mut hi: f64| {
        let positive = drift(lo) > 0.0;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if (drift(mid) > 0.0) == positive { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    };
    let tau = (0..period)
        .find_map(|i| {
            let (lo, hi) = (i as f64, i as f64 + 1.0);
            let (d_lo, d_hi) = (drift(lo), drift(hi));
            if d_lo == 0.0 {
                Some(lo)
            } else if (d_lo > 0.0) != (d_hi > 0.0) {
                Some(bisect(lo, hi))
            } else {
                None
            }
        })
        .unwrap_or(0.0);
    let lines: Vec<(usize, f64, f64)> = lines.iter().map(|&(h, a, phi)| (h, a, phi + theta(h) * tau)).collect();

    Ok((0..n_samples)
        .map(|k| {
            // θ_h (k + 1/2) = π (h (2k + 1) mod 2N) / N: the argument stays small, so that every
            // period is the same
            let value: f64 = lines
                .iter()
                .map(|&(h, a, psi)| a * (std::f64::consts::PI * ((h * (2 * k + 1)) % (2 * period)) as f64 / period as f64 + psi).sin())
                .sum();
            T::from(scale * value).unwrap()
        })
        .collect())
}

/// `multisine` with the amplitudes `1 / |G(j ω_h)|` of the model `model` (e.g. a
/// `TransferFunction`, a `TransferFunctionWithDelay` or a `FrequencyTransferFunction`) at the
/// harmonics `ω_h = 2π h f0`: the output spectrum of the model is flat, so that the measurement
/// noise does not bury the frequencies where the gain is low (e.g. the high frequencies, where it
/// falls like `1 / f^2` or faster).
///
/// Below the harmonic `flat_below` the amplitude is held at its value there (`1 / |G(j ω_flat_below)|`),
/// e.g. so that the low frequencies of a plant with integrators, where `1 / |G|` vanishes, are still
/// excited; 0 (or the lowest harmonic) shapes every line.
///
/// ```
/// use dsmc::{tf, signal::excitation::flat_output_multisine};
///
/// // Rigid body 100 / s^2, 1 ..= 300 Hz (fundamental 1 Hz at 10 kHz), 4 s, shaped from 15 Hz on
/// let harmonics: Vec<usize> = (1..=300).collect();
/// let u: Vec<f64> = flat_output_multisine(&tf!("100 / s^2"), 4.0, 1e-4, 1.0, &harmonics, 15, 1).unwrap();
/// assert_eq!(u.len(), 40000);
/// ```
pub fn flat_output_multisine<T: Float + FloatConst + 'static>(
    model: impl Into<FrequencyTransferFunction<T>>,
    tlen: T,
    ts: T,
    fundamental_frequency: T,
    harmonics: &[usize],
    flat_below: usize,
    seed: u64,
) -> Result<Vec<T>, ExcitationError> {
    let model = model.into();
    let omega = |h: usize| T::from(2.0 * std::f64::consts::PI * h as f64).unwrap() * fundamental_frequency;
    multisine(tlen, ts, fundamental_frequency, harmonics, |h| T::one() / model.response(omega(h.max(flat_below))).norm(), seed)
}

/// Error of a duration `duration / ts` that is not a whole number of samples.
fn duration_error<T: Float>(kind: DurationError, duration: T, ts: T) -> ExcitationError {
    let (duration, ts) = (duration.to_f64().unwrap_or(f64::NAN), ts.to_f64().unwrap_or(f64::NAN));
    match kind {
        DurationError::Invalid => ExcitationError::InvalidDuration { duration, ts },
        DurationError::Fractional => ExcitationError::FractionalDuration { duration, ts },
    }
}

/// The harmonics of a multisine of `period` samples must be distinct (the RMS adds up the powers of
/// distinct lines) and within `1 <= h < period / 2`: `h = 0` is a constant, `h = period / 2` a line
/// sampled at fixed points of its cycle (zero or constant power, depending on the phase), and
/// `h > period / 2` an alias of `period - h`.
fn check_harmonics(period: usize, harmonics: &[usize]) -> Result<(), ExcitationError> {
    for (i, &h) in harmonics.iter().enumerate() {
        if h == 0 || 2 * h >= period {
            return Err(ExcitationError::InvalidHarmonic { harmonic: h, samples_per_period: period });
        }
        if harmonics[..i].contains(&h) {
            return Err(ExcitationError::DuplicateHarmonic { harmonic: h });
        }
    }
    Ok(())
}

/// Invalid arguments of the excitation signals.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ExcitationError {
    #[error("duration {duration} s is negative or not finite, or sampling period {ts} s not positive and finite")]
    InvalidDuration { duration: f64, ts: f64 },
    #[error("duration {duration} s is not a whole number of sampling periods {ts} s")]
    FractionalDuration { duration: f64, ts: f64 },
    #[error("fundamental frequency {fundamental_frequency} Hz is not positive and finite, or sampling period {ts} s not positive and finite")]
    InvalidFundamental { fundamental_frequency: f64, ts: f64 },
    #[error("the period of the fundamental frequency {fundamental_frequency} Hz is not a whole number of sampling periods {ts} s")]
    FractionalPeriod { fundamental_frequency: f64, ts: f64 },
    #[error("harmonic {harmonic} is not within 1 <= h < N / 2 (N = {samples_per_period} samples per period)")]
    InvalidHarmonic { harmonic: usize, samples_per_period: usize },
    #[error("harmonic {harmonic} is given more than once")]
    DuplicateHarmonic { harmonic: usize },
    #[error("the amplitude of harmonic {harmonic} is zero or not finite")]
    InvalidAmplitude { harmonic: usize },
    #[error("no harmonics, or their total power is not finite")]
    NoPower,
}

/// Linear chirp (swept sine) of unit amplitude from `f0` to `f1` \[Hz\] over `tlen` \[s\] sampled
/// with period `ts` \[s\] (`tlen` a whole number of samples):
///
/// ```text
/// u[k] = sin(2π (f0 t + (f1 - f0) t^2 / (2 tlen))),   t = k ts
/// ```
///
/// Not periodic: validate with `Validation::coherence_test` / `cross_correlation`, not `line_test`.
pub fn chirp<T: Float>(tlen: T, ts: T, f0: T, f1: T) -> Result<Vec<T>, ExcitationError> {
    let n_samples = whole_samples(tlen, ts).map_err(|kind| duration_error(kind, tlen, ts))?;
    let (ts, f0, f1) = (ts.to_f64().unwrap(), f0.to_f64().unwrap(), f1.to_f64().unwrap());
    let duration = n_samples as f64 * ts;
    Ok((0..n_samples)
        .map(|k| {
            let t = k as f64 * ts;
            T::from((2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) * t * t / (2.0 * duration))).sin()).unwrap()
        })
        .collect())
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
    use crate::TransferFunction;

    /// DFT of one period at harmonic `h`, as an amplitude (`|U| 2 / period`).
    fn line_amplitude(u: &[f64], period: usize, h: usize) -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for (k, v) in u[..period].iter().enumerate() {
            let theta = 2.0 * std::f64::consts::PI * (h * k % period) as f64 / period as f64;
            re += v * theta.cos();
            im -= v * theta.sin();
        }
        2.0 * (re * re + im * im).sqrt() / period as f64
    }

    #[test]
    fn multisine_has_the_given_amplitudes() {
        // Fundamental 1 Hz at 1 kHz: 1000 samples per period
        let (period, harmonics) = (1000, (5..=200).step_by(5).collect::<Vec<usize>>());
        let u: Vec<f64> = multisine(3.0, 1e-3, 1.0, &harmonics, |h| 1.0 / h as f64, 7).unwrap();
        assert_eq!(u.len(), 3 * period);
        assert!((0..2 * period).all(|k| (u[k] - u[k + period]).abs() < 1e-12));
        let rms = (u[..period].iter().map(|v| v * v).sum::<f64>() / period as f64).sqrt();
        assert!((rms - 1.0).abs() < 1e-12, "rms {rms}");
        // Line amplitudes c / h, nothing elsewhere
        let c = 1.0 / (harmonics.iter().map(|&h| 1.0 / (h * h) as f64).sum::<f64>() / 2.0).sqrt();
        for h in 1..period / 2 {
            let expected = if harmonics.contains(&h) { c / h as f64 } else { 0.0 };
            assert!((line_amplitude(&u, period, h) - expected).abs() < 1e-12, "harmonic {h}");
        }
        // Reproducible from the seed, different for another seed
        assert_eq!(u, multisine::<f64>(3.0, 1e-3, 1.0, &harmonics, |h| 1.0 / h as f64, 7).unwrap());
        assert_ne!(u, multisine::<f64>(3.0, 1e-3, 1.0, &harmonics, |h| 1.0 / h as f64, 8).unwrap());

        // Equal amplitudes at a fundamental of 10 Hz: 100 samples per period, lines at 10 h Hz
        let u: Vec<f64> = multisine(0.3, 1e-3, 10.0, &[1, 2, 3], |_| 1.0, 1).unwrap();
        assert_eq!(u.len(), 300);
        for h in 1..50 {
            let expected = if h <= 3 { (2.0f64 / 3.0).sqrt() } else { 0.0 };
            assert!((line_amplitude(&u, 100, h) - expected).abs() < 1e-12, "harmonic {h}");
        }
    }

    #[test]
    fn multisine_does_not_drift_through_integrators() {
        // Double accumulator over whole periods: the velocity sums to zero over a period (no
        // ramp of the position), with low lines that a random phase would make drift
        let (period, harmonics) = (1000, (1..=20).collect::<Vec<usize>>());
        let u: Vec<f64> = multisine(10.0, 1e-3, 1.0, &harmonics, |_| 1.0, 3).unwrap();
        let (mut v, mut x) = (0.0, 0.0);
        let mut positions = vec![];
        for (k, &uk) in u.iter().enumerate() {
            v += uk;
            x += v;
            if (k + 1) % period == 0 {
                assert!(v.abs() < 1e-9, "velocity {v} after {} periods", (k + 1) / period);
                positions.push(x);
            }
        }
        assert!(positions.windows(2).all(|p| (p[1] - p[0]).abs() < 1e-6), "{positions:?}");
    }

    #[test]
    fn multisine_does_not_drift_through_a_sampled_double_integrator() {
        // 1 / s^2 behind a zero-order hold: the position at the end of each period stays the
        // same (no ramp), from the very first period on
        use crate::{tf, DiscreteSystem, discretize::Zoh};
        let (ts, period) = (1e-3, 1000);
        for seed in 0..5 {
            let u: Vec<f64> = multisine(6.0, ts, 1.0, &(1..=30).collect::<Vec<usize>>(), |h| h as f64, seed).unwrap();
            let mut plant = DiscreteSystem::try_from(&tf!("1 / s^2").discretize(Zoh, ts).unwrap()).unwrap();
            let x: Vec<f64> = u.iter().map(|&uk| plant.update(uk)).collect();
            let ends: Vec<f64> = (1..=6).map(|p| x[p * period - 1]).collect();
            let range = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(ends.windows(2).all(|e| (e[1] - e[0]).abs() < 1e-9 * range), "seed {seed}: {ends:?}");
        }
    }

    #[test]
    fn flat_output_multisine_flattens_the_output() {
        use crate::tf;
        let (ts, period) = (1e-3, 1000);
        let g = tf!("1000 / (s^2 + 20 s + 1000)");
        let harmonics: Vec<usize> = (1..=100).collect();
        let u: Vec<f64> = flat_output_multisine(&g, 1.0, ts, 1.0, &harmonics, 10, 5).unwrap();
        let g_jw = g.frequency_transfer_function();
        let gain = |h: usize| g_jw.response(2.0 * std::f64::consts::PI * h as f64).norm();
        // |U_h| |G_h| is the same from harmonic 10 on, |U_h| the same below
        let output = line_amplitude(&u, period, 10) * gain(10);
        for &h in &harmonics {
            let a = line_amplitude(&u, period, h);
            if h >= 10 {
                assert!((a * gain(h) - output).abs() < 1e-9 * output, "harmonic {h}");
            } else {
                assert!((a - line_amplitude(&u, period, 10)).abs() < 1e-12, "harmonic {h}");
            }
        }
    }

    #[test]
    fn multisine_rejects_invalid_arguments() {
        // Fundamental 10 Hz at 1 kHz: 100 samples per period
        let ms = |harmonics: &[usize], amplitude: f64| multisine::<f64>(0.1, 1e-3, 10.0, harmonics, |_| amplitude, 1);
        let invalid = |harmonic| Err(ExcitationError::InvalidHarmonic { harmonic, samples_per_period: 100 });
        assert_eq!(ms(&[0, 1], 1.0), invalid(0));
        assert_eq!(ms(&[1, 50], 1.0), invalid(50));
        assert_eq!(ms(&[1, 70], 1.0), invalid(70));
        assert_eq!(ms(&[1, 2, 1], 1.0), Err(ExcitationError::DuplicateHarmonic { harmonic: 1 }));
        assert_eq!(ms(&[3], 0.0), Err(ExcitationError::InvalidAmplitude { harmonic: 3 }));
        assert_eq!(ms(&[3], f64::NAN), Err(ExcitationError::InvalidAmplitude { harmonic: 3 }));
        assert_eq!(ms(&[], 1.0), Err(ExcitationError::NoPower));
        assert_eq!(ms(&[1, 2], 1e200), Err(ExcitationError::NoPower));
        assert!(ms(&[1, 49], 1.0).is_ok());
        // An undamped resonance on a line (harmonic 2 of 10 Hz = 20 Hz): the amplitude 1 / |G| is
        // zero there
        let resonance = TransferFunction::continuous(&[1.0], &[1.0, 0.0, (2.0 * std::f64::consts::PI * 20.0).powi(2)]);
        assert_eq!(flat_output_multisine::<f64>(&resonance, 0.1, 1e-3, 10.0, &[1, 2, 3], 0, 1), Err(ExcitationError::InvalidAmplitude { harmonic: 2 }));
        assert!(flat_output_multisine::<f64>(&resonance, 0.1, 1e-3, 10.0, &[1, 3], 0, 1).is_ok());
    }

    #[test]
    fn durations_and_fundamental_must_be_whole_samples() {
        let ms = |tlen: f64, f0: f64| multisine::<f64>(tlen, 1e-3, f0, &[1], |_| 1.0, 1);
        assert_eq!(ms(1.0005, 10.0), Err(ExcitationError::FractionalDuration { duration: 1.0005, ts: 1e-3 }));
        assert!(matches!(ms(-1.0, 10.0), Err(ExcitationError::InvalidDuration { .. })));
        // 3 Hz at 1 kHz: 333.3 samples per period
        assert_eq!(ms(1.0, 3.0), Err(ExcitationError::FractionalPeriod { fundamental_frequency: 3.0, ts: 1e-3 }));
        for f0 in [0.0, -1.0, f64::NAN] {
            assert!(matches!(ms(1.0, f0), Err(ExcitationError::InvalidFundamental { .. })), "{f0}");
        }
        assert!(matches!(multisine::<f64>(1.0, 0.0, 10.0, &[1], |_| 1.0, 1), Err(ExcitationError::InvalidDuration { .. })));
        assert_eq!(chirp::<f64>(1.0005, 1e-3, 1.0, 2.0), Err(ExcitationError::FractionalDuration { duration: 1.0005, ts: 1e-3 }));
        // A length of zero gives no samples; f32: 1 / (20 * 1e-4) = 500 within the rounding
        assert_eq!(ms(0.0, 10.0), Ok(vec![]));
        assert_eq!(multisine::<f32>(0.1, 1e-4, 20.0, &[1, 2], |_| 1.0, 1).unwrap().len(), 1000);
    }

    #[test]
    fn chirp_sweeps_linearly() {
        let u: Vec<f64> = chirp(1.0, 1e-3, 1.0, 11.0).unwrap();
        assert_eq!(u.len(), 1000);
        assert!(u[0].abs() < 1e-12 && u.iter().all(|v| v.abs() <= 1.0));
        // Instantaneous frequency f0 + (f1 - f0) t / T: 6 Hz in the middle, i.e. zero crossings
        // about 1 / 12 s apart there
        let crossings: Vec<usize> = (450..560).filter(|&k| u[k] <= 0.0 && u[k + 1] > 0.0 || u[k] >= 0.0 && u[k + 1] < 0.0).collect();
        let spacing = (crossings[crossings.len() - 1] - crossings[0]) as f64 / (crossings.len() - 1) as f64 * 1e-3;
        assert!((spacing - 1.0 / 12.0).abs() < 0.01, "spacing {spacing}");
    }
}
