//! Multisines: sums of harmonics of a fundamental frequency with random phases.

use num_traits::{Float, FloatConst};

use super::{duration_error, ExcitationError};
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
        let phase = 2.0 * std::f64::consts::PI * uniform(&mut state);
        lines.push((h, a, phase));
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

/// `multisine` with the amplitudes `|F(j ω_h)|` of the filter `filter` (e.g. a `TransferFunction`,
/// a `TransferFunctionWithDelay` or a `FrequencyTransferFunction`) at the harmonics
/// `ω_h = 2π h f0`, scaled to unit RMS like `multisine`: the band is that of `harmonics`, the
/// shape of the spectrum within it that of `|F|` (its phase is not used).
///
/// ```
/// use dsmc::{tf, signal::excitation::shaped_multisine};
///
/// // First-order roll-off from 50 Hz, 1 ..= 300 Hz (fundamental 1 Hz at 10 kHz), 4 s
/// let wc = 2.0 * std::f64::consts::PI * 50.0;
/// let harmonics: Vec<usize> = (1..=300).collect();
/// let u: Vec<f64> = shaped_multisine(&tf!("{wc} / (s + {wc})"), 4.0, 1e-4, 1.0, &harmonics, 1).unwrap();
/// assert_eq!(u.len(), 40000);
/// ```
pub fn shaped_multisine<T: Float + FloatConst + 'static>(
    filter: impl Into<FrequencyTransferFunction<T>>,
    tlen: T,
    ts: T,
    fundamental_frequency: T,
    harmonics: &[usize],
    seed: u64,
) -> Result<Vec<T>, ExcitationError> {
    let filter = filter.into();
    let omega = |h: usize| T::from(2.0 * std::f64::consts::PI * h as f64).unwrap() * fundamental_frequency;
    multisine(tlen, ts, fundamental_frequency, harmonics, |h| filter.response(omega(h)).norm(), seed)
}

/// `shaped_multisine` with the filter `1 / G` of the model `model`: the amplitudes are
/// `1 / |G(j ω_h)|`, so that the output spectrum of the model is flat and the measurement noise
/// does not bury the frequencies where the gain is low (e.g. the high frequencies, where it falls
/// like `1 / f^2` or faster).
///
/// The band is that of `harmonics`. Every line is shaped, so the input lines of a plant with
/// integrators are small at the low frequencies (where `1 / |G|` is); for another shape, e.g. a
/// lower bound on the input amplitude, give `multisine` the amplitudes directly.
///
/// ```
/// use dsmc::{tf, signal::excitation::flat_output_multisine};
///
/// // Rigid body 100 / s^2, 10 ..= 300 Hz (fundamental 1 Hz at 10 kHz), 4 s
/// let harmonics: Vec<usize> = (10..=300).collect();
/// let u: Vec<f64> = flat_output_multisine(&tf!("100 / s^2"), 4.0, 1e-4, 1.0, &harmonics, 1).unwrap();
/// assert_eq!(u.len(), 40000);
/// ```
pub fn flat_output_multisine<T: Float + FloatConst + 'static>(
    model: impl Into<FrequencyTransferFunction<T>>,
    tlen: T,
    ts: T,
    fundamental_frequency: T,
    harmonics: &[usize],
    seed: u64,
) -> Result<Vec<T>, ExcitationError> {
    shaped_multisine(model.into().inverse(), tlen, ts, fundamental_frequency, harmonics, seed)
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

/// splitmix64: uniform in [0, 1).
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}
