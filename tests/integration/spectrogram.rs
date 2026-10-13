use std::f64::consts::PI;

use dsmc::fft::{power_spectral_density, spectrogram};

#[test]
fn spectrogram_has_the_segments_of_the_window_time() {
    let (ts, n, window_time) = (1e-3, 1000, 0.1);
    let data: Vec<f64> = (0..n).map(|k| (k as f64 * 0.37).sin()).collect();
    let s = spectrogram(&data, ts, window_time);

    // L = 100, step 50: segments start at 0, 50, ..., 900
    assert_eq!(s.omega.len(), 51);
    assert_eq!(s.time.len(), 19);
    assert_eq!(s.psd.len(), 19);
    assert!(s.psd.iter().all(|row| row.len() == 51));
    assert!((s.time[0] - 0.05).abs() < 1e-12);
    assert!((s.time[18] - 0.95).abs() < 1e-12);
    assert!((s.omega[1] - 2.0 * PI * 10.0).abs() < 1e-9);
}

#[test]
fn spectrogram_follows_a_change_of_frequency() {
    // 50 Hz for the first second, 200 Hz for the next
    let (ts, n, a) = (1e-3, 2000, 2.0);
    let data: Vec<f64> = (0..n)
        .map(|k| {
            let t = k as f64 * ts;
            let f = if t < 1.0 { 50.0 } else { 200.0 };
            a * (2.0 * PI * f * t).sin()
        })
        .collect();
    let window_time = 0.1;
    let s = spectrogram(&data, ts, window_time);
    let df = 1.0 / window_time;

    for (t, row) in s.time.iter().zip(&s.psd) {
        // skip the segments that straddle the switch
        if (t - 1.0).abs() < window_time {
            continue;
        }
        let expected = if *t < 1.0 { 50.0 } else { 200.0 };
        let peak = (0..row.len()).max_by(|&i, &j| row[i].total_cmp(&row[j])).unwrap();
        assert!((s.omega[peak] / (2.0 * PI) - expected).abs() <= df, "t = {t}");

        // each segment integrates to the variance of the sinusoid
        let power: f64 = row.iter().map(|p| p * df).sum();
        assert!((power - a * a / 2.0).abs() < 0.02 * a * a / 2.0, "t = {t}: {power}");
    }
}

#[test]
fn spectrogram_averages_to_the_psd_with_the_same_segments() {
    // 1024 samples, window 128 samples: power_spectral_density with 8 segments uses the same ones
    let ts = 1e-3;
    let data: Vec<f64> = (0..1024).map(|k| (k as f64 * 0.3).sin() + 0.2 * (k as f64 * 1.9).cos()).collect();
    let s = spectrogram(&data, ts, 128.0 * ts);
    let psd = power_spectral_density(&data, ts, 8);

    assert_eq!(s.omega.len(), psd.len());
    for (k, p) in psd.iter().enumerate() {
        assert_eq!(s.omega[k], p.omega);
        let mean = s.psd.iter().map(|row| row[k]).sum::<f64>() / s.psd.len() as f64;
        assert!((mean - p.value).abs() <= 1e-12 * p.value.abs().max(1.0), "bin {k}");
    }
}

#[test]
fn spectrogram_clamps_the_window_to_the_record() {
    let data = [1.0, -1.0, 2.0];
    let s = spectrogram(&data, 1.0, 10.0);
    assert_eq!(s.time.len(), 1);
    assert_eq!(s.omega.len(), 2);

    assert!(spectrogram::<f64>(&[1.0], 1e-3, 0.1).psd.is_empty());
}

#[test]
#[should_panic(expected = "window_time")]
fn spectrogram_rejects_a_non_positive_window() {
    spectrogram(&[0.0; 10], 1e-3, 0.0);
}

#[test]
fn periodic_hann_leaks_a_sinusoid_on_a_bin_into_its_neighbours_only() {
    // 100 Hz on bin 10 of a 100-sample window at 1 kHz: the periodic Hann window spreads it over
    // bins 9, 10, 11 with powers 1 : 4 : 1, and nothing else
    let (ts, n) = (1e-3, 1000);
    let data: Vec<f64> = (0..n).map(|k| (2.0 * PI * 100.0 * k as f64 * ts).sin()).collect();
    let s = spectrogram(&data, ts, 0.1);

    for row in &s.psd {
        let peak = row[10];
        assert!((row[9] / peak - 0.25).abs() < 1e-9 && (row[11] / peak - 0.25).abs() < 1e-9);
        for (k, p) in row.iter().enumerate() {
            if !(9..=11).contains(&k) {
                assert!(*p < 1e-12 * peak, "bin {k}: {p}");
            }
        }
    }
}
