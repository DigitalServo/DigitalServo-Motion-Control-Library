//! Excitation signals (`dsmc::signal::excitation`).

use dsmc::signal::excitation::*;
use dsmc::{FrequencyTransferFunction, TransferFunction};

// --- multisine ---

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
    use dsmc::{tf, DiscreteSystem, discretize::Zoh};
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
    use dsmc::tf;
    let (ts, period) = (1e-3, 1000);
    let g = tf!("1000 / (s^2 + 20 s + 1000)");
    let harmonics: Vec<usize> = (1..=100).collect();
    let u: Vec<f64> = flat_output_multisine(&g, 1.0, ts, 1.0, &harmonics, 5).unwrap();
    let g_jw = g.frequency_transfer_function();
    let gain = |h: usize| g_jw.response(2.0 * std::f64::consts::PI * h as f64).norm();
    // |U_h| |G_h| is the same at every line
    let output = line_amplitude(&u, period, 1) * gain(1);
    for &h in &harmonics {
        let a = line_amplitude(&u, period, h);
        assert!((a * gain(h) - output).abs() < 1e-9 * output, "harmonic {h}");
    }
}

#[test]
fn shaped_multisine_takes_the_gain_of_the_filter() {
    use num_complex::Complex;
    // Fundamental 1 Hz at 1 kHz; gain 1 + h on harmonic h, phase ignored
    let filter = FrequencyTransferFunction::new(|w: f64| Complex::new(0.0, 1.0 + w / (2.0 * std::f64::consts::PI)));
    let harmonics: Vec<usize> = (1..=50).collect();
    let shaped: Vec<f64> = shaped_multisine(&filter, 2.0, 1e-3, 1.0, &harmonics, 3).unwrap();
    let direct: Vec<f64> = multisine(2.0, 1e-3, 1.0, &harmonics, |h| 1.0 + h as f64, 3).unwrap();
    assert!(shaped.iter().zip(&direct).all(|(a, b)| (a - b).abs() < 1e-12));
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
    assert_eq!(flat_output_multisine::<f64>(&resonance, 0.1, 1e-3, 10.0, &[1, 2, 3], 1), Err(ExcitationError::InvalidAmplitude { harmonic: 2 }));
    assert!(flat_output_multisine::<f64>(&resonance, 0.1, 1e-3, 10.0, &[1, 3], 1).is_ok());
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
    // A length of zero gives no samples; f32: 1 / (20 * 1e-4) = 500 within the rounding
    assert_eq!(ms(0.0, 10.0), Ok(vec![]));
    assert_eq!(multisine::<f32>(0.1, 1e-4, 20.0, &[1, 2], |_| 1.0, 1).unwrap().len(), 1000);
}

// --- chirp ---

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

#[test]
fn chirp_duration_must_be_whole_samples() {
    assert_eq!(chirp::<f64>(1.0005, 1e-3, 1.0, 2.0), Err(ExcitationError::FractionalDuration { duration: 1.0005, ts: 1e-3 }));
}

#[test]
fn shaped_chirp_follows_the_gain_at_the_instantaneous_frequency() {
    use num_complex::Complex;
    let rms = |u: &[f64]| (u.iter().map(|v| v * v).sum::<f64>() / u.len() as f64).sqrt();
    // A constant gain: the chirp scaled to unit RMS
    let constant = FrequencyTransferFunction::new(|_: f64| Complex::new(0.0, -3.0));
    let u: Vec<f64> = shaped_chirp(&constant, 1.0, 1e-3, 1.0, 11.0).unwrap();
    let plain: Vec<f64> = chirp(1.0, 1e-3, 1.0, 11.0).unwrap();
    assert!((rms(&u) - 1.0).abs() < 1e-12);
    assert!(u.iter().zip(&plain).all(|(a, b)| (a - b / rms(&plain)).abs() < 1e-12));
    // Gain f [Hz]: the envelope grows with the instantaneous frequency f0 + (f1 - f0) t / tlen
    let ramp = FrequencyTransferFunction::new(|w: f64| Complex::new(w / (2.0 * std::f64::consts::PI), 0.0));
    let u: Vec<f64> = shaped_chirp(&ramp, 1.0, 1e-3, 1.0, 11.0).unwrap();
    let ratio = |k: usize| u[k] / plain[k] / (1.0 + 10.0 * k as f64 * 1e-3);
    assert!((0..1000).filter(|&k| plain[k].abs() > 0.1).all(|k| (ratio(k) - ratio(500)).abs() < 1e-9 * ratio(500).abs()));
    // No power
    let zero = FrequencyTransferFunction::new(|_: f64| Complex::new(0.0, 0.0));
    assert_eq!(shaped_chirp(&zero, 1.0, 1e-3, 1.0, 11.0), Err(ExcitationError::NoPower));
    assert_eq!(shaped_chirp(&constant, 0.0, 1e-3, 1.0, 11.0), Err(ExcitationError::NoPower));
}
