use std::f64::consts::PI;

use dsmc::fft::{amplitude_spectrum, power_spectral_density, welch};

/// Uniform samples in `[-0.5, 0.5)` from a fixed-seed LCG.
fn uniform_noise(n: usize, mut seed: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

#[test]
fn amplitude_spectrum_reads_the_amplitudes_of_a_sinusoid_and_dc() {
    let (ts, n) = (1e-3, 1000);
    let data: Vec<f64> = (0..n)
        .map(|k| 0.7 + 2.5 * (2.0 * PI * 50.0 * k as f64 * ts).sin())
        .collect();
    let s = amplitude_spectrum(&data, ts);

    // bins 0..=N/2, 1 Hz apart
    assert_eq!(s.len(), n / 2 + 1);
    assert!((s[500].omega - 2.0 * PI * 500.0).abs() < 1e-9);
    assert!((s[50].omega - 2.0 * PI * 50.0).abs() < 1e-9);

    assert!((s[50].value - 2.5).abs() < 1e-9, "{}", s[50].value);
    assert!((s[0].value - 0.7).abs() < 1e-9, "{}", s[0].value);
    assert!(s[49].value < 1e-9, "{}", s[49].value);
    assert!(s[51].value < 1e-9, "{}", s[51].value);
}

#[test]
fn amplitude_spectrum_does_not_double_the_nyquist_bin_of_an_even_length() {
    // (-1)^k c lies entirely at the Nyquist bin k = N/2
    let (ts, n, c) = (0.5, 8, 1.5);
    let data: Vec<f64> = (0..n).map(|k| if k % 2 == 0 { c } else { -c }).collect();
    let s = amplitude_spectrum(&data, ts);

    assert_eq!(s.len(), n / 2 + 1);
    assert!((s[n / 2].omega - PI / ts).abs() < 1e-12);
    assert!((s[n / 2].value - c).abs() < 1e-12, "{}", s[n / 2].value);
    for p in &s[..n / 2] {
        assert!(p.value < 1e-12);
    }
}

#[test]
fn amplitude_spectrum_doubles_the_last_bin_of_an_odd_length() {
    // with N odd, the last bin k = (N-1)/2 is below the Nyquist frequency and has a mirror image
    let (ts, n, a) = (0.1, 9, 0.8);
    let data: Vec<f64> = (0..n).map(|k| a * (2.0 * PI * 4.0 * k as f64 / n as f64).cos()).collect();
    let s = amplitude_spectrum(&data, ts);

    assert_eq!(s.len(), 5);
    assert!((s[4].omega - 2.0 * PI * 4.0 / (n as f64 * ts)).abs() < 1e-12);
    assert!((s[4].value - a).abs() < 1e-12, "{}", s[4].value);
    for p in &s[..4] {
        assert!(p.value < 1e-12);
    }
}

#[test]
fn amplitude_spectrum_of_an_empty_record_is_empty() {
    assert!(amplitude_spectrum::<f64>(&[], 1e-3).is_empty());
}

#[test]
fn psd_of_a_sinusoid_integrates_to_its_variance() {
    let (ts, n, a, f) = (1e-3, 10000, 2.0, 50.0);
    let data: Vec<f64> = (0..n)
        .map(|k| 0.5 + a * (2.0 * PI * f * k as f64 * ts).sin())
        .collect();
    let psd = power_spectral_density(&data, ts, 8);

    let nperseg = 2 * (psd.len() - 1);
    let df = 1.0 / (ts * nperseg as f64);
    let power: f64 = psd.iter().map(|p| p.value * df).sum();
    assert!((power - a * a / 2.0).abs() < 0.02 * a * a / 2.0, "{power}");

    let peak = psd.iter().max_by(|p, q| p.value.total_cmp(&q.value)).unwrap();
    assert!((peak.omega / (2.0 * PI) - f).abs() <= df, "{}", peak.omega / (2.0 * PI));
}

#[test]
fn psd_of_white_noise_is_twice_the_variance_times_ts() {
    let (ts, n) = (1e-4, 1 << 16);
    let sigma2 = 1.0 / 12.0;
    let data = uniform_noise(n, 2024);
    let psd = power_spectral_density(&data, ts, 64);

    // inner bins only: DC is removed with the mean, and DC and Nyquist are not doubled
    let inner = &psd[1..psd.len() - 1];
    let mean = inner.iter().map(|p| p.value).sum::<f64>() / inner.len() as f64;
    let expected = 2.0 * sigma2 * ts;
    assert!((mean - expected).abs() < 0.03 * expected, "{mean} vs {expected}");
}

#[test]
fn psd_has_the_frequencies_of_welch() {
    let (ts, n) = (1e-3, 3000);
    let u = uniform_noise(n, 7);
    let y = uniform_noise(n, 8);
    for n_segments in [1, 3, 8, 100] {
        let psd = power_spectral_density(&u, ts, n_segments);
        let (g, _) = welch(&u, &y, ts, n_segments);
        assert_eq!(psd.len(), g.len());
        for (p, r) in psd.iter().zip(&g) {
            assert_eq!(p.omega, r.omega);
        }
    }
}

#[test]
fn psd_of_fewer_than_two_samples_is_empty() {
    assert!(power_spectral_density::<f64>(&[], 1e-3, 4).is_empty());
    assert!(power_spectral_density(&[1.0], 1e-3, 4).is_empty());
}
