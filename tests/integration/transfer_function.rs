#[test]
fn test_closed_tf() {
    use dsmc::tf;
    let tf_open: dsmc::TransferFunction<f64> = tf!("(-2s + 10) / (s^2 + s + 10)");
    let tf_closed = tf_open.unity_feedback();
    println!("{}", tf_closed);
    println!("{}", tf_closed.pz_map());

    // L / (1 + L) = (-2s + 10) / (s^2 - s + 20)
    let expected = tf!("(-2s + 10) / (s^2 - s + 20)");
    assert_eq!(tf_closed.numerator, expected.numerator);
    assert_eq!(tf_closed.denominator, expected.denominator);

    // Same result via `+` / `/` (up to the overall scale of numerator and denominator)
    let via_ops = &tf_open / &(1.0 + &tf_open);
    let scale = via_ops.denominator[0];
    for (a, b) in via_ops.numerator.iter().zip(expected.numerator.iter()) {
        assert!((a / scale - b).abs() < 1e-9);
    }
    for (a, b) in via_ops.denominator.iter().zip(expected.denominator.iter()) {
        assert!((a / scale - b).abs() < 1e-9);
    }
}

/// Dead time in samples: whole numbers (within the rounding of f32 too), and errors instead of a
/// panic for a sampling period that is not positive, or a delay that is not finite.
#[test]
fn test_delay_samples() {
    use dsmc::{SimulationError, TransferFunction, TransferFunctionWithDelay};
    let delayed = |delay: f64| TransferFunctionWithDelay::new(TransferFunction::continuous(&[1.0], &[1.0, 1.0]), delay);
    assert_eq!(delayed(3e-3).delay_samples(1e-3), Ok(3));
    assert_eq!(delayed(0.0).delay_samples(1e-3), Ok(0));
    assert_eq!(TransferFunctionWithDelay::new(TransferFunction::continuous(&[1.0f32], &[1.0, 1.0]), 0.05).delay_samples(1e-4), Ok(500));
    assert!(matches!(delayed(3.5e-3).delay_samples(1e-3), Err(SimulationError::FractionalDelay { .. })));
    assert!(matches!(delayed(-3e-3).delay_samples(1e-3), Err(SimulationError::NegativeDelay { .. })));
    assert!(matches!(delayed(f64::NAN).delay_samples(1e-3), Err(SimulationError::FractionalDelay { .. })));
    for ts in [0.0, -1e-3, f64::NAN] {
        assert!(matches!(delayed(3e-3).delay_samples(ts), Err(SimulationError::InvalidSamplingPeriod { .. })), "{ts}");
    }
}
