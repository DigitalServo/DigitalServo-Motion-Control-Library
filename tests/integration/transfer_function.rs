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

mod negative_powers {
    use dsmc::{Continuous, Discrete, Polynomial, TransferFunction};

    const TOL: f64 = 1e-12;

    /// Coefficients (numerator, denominator) without leading zeros, scaled so that the
    /// denominator is monic.
    fn normalized<D: dsmc::Domain>(tf: &TransferFunction<f64, D>) -> (Vec<f64>, Vec<f64>) {
        let trim = |p: &[f64]| p.iter().copied().skip_while(|c| *c == 0.0).collect::<Vec<_>>();
        let num = trim(&tf.numerator.0);
        let den = trim(&tf.denominator.0);
        let scale = den[0];
        (num.iter().map(|c| c / scale).collect(), den.iter().map(|c| c / scale).collect())
    }

    fn assert_coeffs(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} != {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < TOL, "{actual:?} != {expected:?}");
        }
    }

    /// 8-tap moving average: (z^7 + ... + 1) / (8 z^7)
    #[test]
    fn parse_moving_average_with_negative_powers() {
        let tf: TransferFunction<f64, Discrete> =
            "(1 + z^-1 + z^-2 + z^-3 + z^-4 + z^-5 + z^-6 + z^-7)/8".parse().unwrap();
        let (num, den) = normalized(&tf);
        assert_coeffs(&num, &[0.125; 8]);
        let mut expected_den = vec![1.0];
        expected_den.extend([0.0; 7]);
        assert_coeffs(&den, &expected_den);
    }

    /// A 32-tap FIR keeps degree 31 (no inflated degree, no leftover common factor).
    #[test]
    fn parse_long_fir_keeps_minimal_degree() {
        let src = format!(
            "({})/32",
            (0..32).map(|k| format!("z^-{k}")).collect::<Vec<_>>().join(" + ")
        );
        let tf: TransferFunction<f64, Discrete> = src.parse().unwrap();
        let (num, den) = normalized(&tf);
        assert_coeffs(&num, &[1.0 / 32.0; 32]);
        assert_eq!(den.len(), 32);
        assert!(den[1..].iter().all(|c| *c == 0.0));
    }

    /// `reduced` alone: a common root of high multiplicity at the origin cancels completely.
    /// z^18 (z^7 + ... + 1) / (8 z^25) -> (z^7 + ... + 1) / (8 z^7)
    #[test]
    fn reduced_cancels_high_multiplicity_root_at_origin() {
        let mut num = vec![1.0; 8];
        num.extend([0.0; 18]);
        let mut den = vec![8.0];
        den.extend([0.0; 25]);
        let tf = TransferFunction::<f64, Discrete>::from_polynomials(Polynomial(num), Polynomial(den)).reduced();
        let (num, den) = normalized(&tf);
        assert_coeffs(&num, &[0.125; 8]);
        assert_eq!(den.len(), 8);
        assert!(den[1..].iter().all(|c| *c == 0.0));
    }

    /// Same in continuous time: s^10 (s + 1) / (s^12 (s + 2)) -> (s + 1) / (s^2 (s + 2))
    #[test]
    fn reduced_cancels_high_multiplicity_root_at_origin_continuous() {
        let tf: TransferFunction<f64, Continuous> = "s^10 (s + 1) / (s^12 (s + 2))".parse().unwrap();
        let (num, den) = normalized(&tf);
        assert_coeffs(&num, &[1.0, 1.0]);
        assert_coeffs(&den, &[1.0, 2.0, 0.0, 0.0]);
    }

    /// Recursive moving average: both the pole-zero cancellation at z = 1 and at the origin.
    /// (1 - z^-4) / (4 (1 - z^-1)) = (z^3 + z^2 + z + 1) / (4 z^3)
    #[test]
    fn parse_recursive_moving_average() {
        let tf: TransferFunction<f64, Discrete> = "(1 - z^-4)/(4(1 - z^-1))".parse().unwrap();
        let (num, den) = normalized(&tf);
        assert_coeffs(&num, &[0.25; 4]);
        assert_coeffs(&den, &[1.0, 0.0, 0.0, 0.0]);
    }

    /// Roots at the origin that are not common remain: z^-2 / (1 - 0.5 z^-1) = 1 / (z (z - 0.5))
    #[test]
    fn parse_keeps_non_common_root_at_origin() {
        let tf: TransferFunction<f64, Discrete> = "z^-2/(1 - 0.5z^-1)".parse().unwrap();
        let (num, den) = normalized(&tf);
        assert_coeffs(&num, &[1.0]);
        assert_coeffs(&den, &[1.0, -0.5, 0.0]);
    }
}
