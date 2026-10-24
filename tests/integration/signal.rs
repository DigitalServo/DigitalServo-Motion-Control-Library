#[test]
fn test_delayer() {
    use dsmc::signal::Delayer;
    let mut delayer = Delayer::new(5);
    for i in 0..20 {
        let out = delayer.output(i);
        // Zero (the default) until the buffer fills, then the input from 5 samples earlier.
        let expected = if i < 5 { 0 } else { i - 5 };
        assert_eq!(out, expected, "sample {i}");
    }
}

/// `detect_period`: a periodic signal with noise and a decaying transient gives its shortest period
/// (not a multiple), noise or a chirp gives `None`.
#[test]
fn test_detect_period() {
    use dsmc::signal::detect_period;
    use dsmc::signal::excitation::multisine;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut gaussian = move || {
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos()
    };
    // Lines 1 ..= 30 of 200 samples, 12 periods, noise 0.3 times the RMS
    let period = 200;
    let lines: Vec<usize> = (1..=30).collect();
    let u: Vec<f64> = multisine(12.0 * period as f64 * 1e-3, 1e-3, 1000.0 / period as f64, &lines, |_| 1.0, 3).unwrap();
    let x: Vec<f64> = u.iter().enumerate().map(|(k, v)| v + 0.3 * gaussian() + 5.0 * (-(k as f64) / 50.0).exp()).collect();
    assert_eq!(detect_period(&x, 1000), Some(period));
    // Even lines only: the signal repeats every 100 samples, and 100 (not 200) is found
    let u: Vec<f64> = multisine(12.0 * period as f64 * 1e-3, 1e-3, 1000.0 / period as f64, &[2, 4, 6], |_| 1.0, 3).unwrap();
    assert_eq!(detect_period(&u, 1000), Some(period / 2));

    let noise: Vec<f64> = (0..4000).map(|_| gaussian()).collect();
    assert_eq!(detect_period(&noise, 1000), None);
    let chirp: Vec<f64> = (0..4000).map(|k| (1e-5 * (k * k) as f64).sin()).collect();
    assert_eq!(detect_period(&chirp, 1000), None);
}
