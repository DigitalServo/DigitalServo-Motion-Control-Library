use dsmc::{Discrete, TransferFunction, dka_method, vieta_formula};

/// Assert that `actual` and `expected` contain the same roots (in any order) within `tol`.
fn assert_roots_close(actual: &[num_complex::Complex<f64>], expected: &[num_complex::Complex<f64>], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "root count: {:?} vs {:?}", actual, expected);
    let mut unused: Vec<_> = actual.to_vec();
    for e in expected {
        let (k, dist) = unused
            .iter()
            .enumerate()
            .map(|(k, a)| (k, (a - e).norm()))
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .unwrap();
        assert!(dist <= tol, "expected root {e} not found in {actual:?} (closest distance {dist:e})");
        unused.remove(k);
    }
}

fn assert_coeffs_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "coefficient count: {:?} vs {:?}", actual, expected);
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tol, "coefficients {actual:?} != {expected:?}");
    }
}

#[test]
fn test_vieta_formula() {
    use num_complex::Complex;
    use dsmc::Polynomial;

    let roots: Vec<Complex<f64>> = vec![
        Complex::new(1.2, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(-2.0, 0.0),
        Complex::new(-1.0, 0.0),
        Complex::new(3.0, 0.0),
    ];

    // (x - 1.2)(x - 2)^2(x + 2)(x + 1)(x - 3)
    let coeffs = vieta_formula(&roots);
    assert!(coeffs.0.iter().all(|c| c.im == 0.0));
    let re: Vec<f64> = coeffs.0.iter().map(|c| c.re).collect();
    assert_coeffs_close(&re, &[1.0, -5.2, 1.8, 25.6, -30.4, -19.2, 28.8], 1e-9);

    // Scaling by a complex constant does not move the roots.
    let coeffs = coeffs.0
        .into_iter()
        .map(|x| Complex{re: 1.2, im: 3.0} * x)
        .collect::<Vec<_>>();
    let coeffs = Polynomial(coeffs);

    // The double root at 2.0 is only found to ~sqrt(eps) accuracy.
    let found = dka_method(&coeffs).unwrap();
    assert_roots_close(&found, &roots, 1e-6);
}

#[test]
fn test_search_roots() {
    use num_complex::Complex;
    use dsmc::Polynomial;

    // s^2 + 20s + 100 = (s + 10)^2
    let coeffs = Polynomial(vec![
        Complex::<f64>::new(1.0, 0.0),
        Complex::<f64>::new(20.0, 0.0),
        Complex::<f64>::new(100.0, 0.0),
    ]);

    let coeffs = coeffs.0
        .into_iter()
        .map(|x| Complex{re: 1.2, im: 3.0} * x)
        .collect::<Vec<_>>();
    let coeffs = Polynomial(coeffs);

    let roots = dka_method(&coeffs).unwrap();
    assert_roots_close(&roots, &[Complex::new(-10.0, 0.0), Complex::new(-10.0, 0.0)], 1e-6);
}


#[test]
fn test_pz_map() {
    use num_complex::Complex;

    let tf = TransferFunction::<f64>::continuous(
        &[1.0, 10.0, 100.0],
        &[1.0, 20.0, 100.0],
    );

    // Zeros: s^2 + 10s + 100 = 0 -> -5 ± j5√3. Poles: (s + 10)^2 = 0 -> -10 (double).
    let pz_map = tf.pz_map();
    let im = 5.0 * 3.0_f64.sqrt();
    assert_roots_close(&pz_map.zeros, &[Complex::new(-5.0, im), Complex::new(-5.0, -im)], 1e-9);
    assert_roots_close(&pz_map.poles, &[Complex::new(-10.0, 0.0), Complex::new(-10.0, 0.0)], 1e-6);

    // Display: the imaginary sign of the double pole is numerical noise, so only check the zeros.
    let text = format!("{:.2}", pz_map);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 6);
    assert_eq!(lines[0], "Poles:");
    assert_eq!(lines[3], "Zeros:");
    assert!(lines[4..].contains(&"  -5.00 + j8.66"));
    assert!(lines[4..].contains(&"  -5.00 - j8.66"));
}

#[test]
fn test_transfer_function_parser() {
    use dsmc::tf;

    let expected = TransferFunction::<f64>::continuous(&[1.0, 10.0, 10.0], &[1.0, 20.0, 100.0]);

    let g = tf!((s^2 + 10.0 * s + 10.0) / (s^2 + 20.0 * s + 100.0));
    assert_eq!(g.numerator, expected.numerator);
    assert_eq!(g.denominator, expected.denominator);

    let g = tf!("(s^2 + 10s + 10) / (s + 10)^2");
    assert_eq!(g.numerator, expected.numerator);
    assert_eq!(g.denominator, expected.denominator);

    // Pole-zero cancellation: (s+1)/((s+1)(s+2)(s+4)) = 1/((s+2)(s+4)) = 1/(s^2 + 6s + 8)
    let g = tf!("(s + 1) / ((s + 1)(s + 2)(s + 4))");
    assert_coeffs_close(&g.numerator, &[1.0], 1e-9);
    assert_coeffs_close(&g.denominator, &[1.0, 6.0, 8.0], 1e-9);
    assert_eq!(format!("{:.2}", g), "1.00 / (s^2 + 6.00 * s + 8.00)");

    // Errors are only observable through `FromStr`, since `tf!` panics.
    assert!("1 / 0".parse::<TransferFunction<f64>>().is_err());
    assert!("(s + 1".parse::<TransferFunction<f64>>().is_err());
    assert!("s^0.5".parse::<TransferFunction<f64>>().is_err());
    assert!("x + 1".parse::<TransferFunction<f64>>().is_err());
}

#[test]
fn test_transfer_function_display() {
    use dsmc::tf;

    let g = tf!("1 / (s + 2)");
    assert_eq!(g.to_string(), "1.0 / (s + 2.0)");

    let g = tf!("(s^2 - 3s) / (2s^2 + 20s + 100)");
    assert_eq!(g.to_string(), "(s^2 - 3.0 * s) / (2.0 * s^2 + 20.0 * s + 100.0)");
    assert_eq!(format!("{:.2}", g), "(s^2 - 3.00 * s) / (2.00 * s^2 + 20.00 * s + 100.00)");

    let g = tf!("-s^2 + 1");
    assert_eq!(g.to_string(), "-s^2 + 1.0");

    // Round trip
    let g = tf!("(s^2 + 10s + 10) / (s^2 + 20s + 100)");
    let h: TransferFunction<f64> = g.to_string().parse().unwrap();
    assert_eq!(g.numerator, h.numerator);
    assert_eq!(g.denominator, h.denominator);
}

#[test]
fn test_discrete_transfer_function() {
    use dsmc::tf;

    // Annotation needed here: `T` falls back to f64 only at the end of type checking,
    // too late for the `.abs()` call on the pole below.
    let g: TransferFunction<f64, Discrete> = tf!("0.5 / (z - 0.5)");
    assert_eq!(g.to_string(), "0.5 / (z - 0.5)");

    let pz = g.pz_map();
    assert!((pz.poles[0].re - 0.5).abs() < 1e-9);

    // `s` is not a valid variable in the z-domain, and vice versa.
    assert!("1 / (s + 1)".parse::<TransferFunction<f64, Discrete>>().is_err());
    assert!("1 / (z + 1)".parse::<TransferFunction<f64>>().is_err());
}

#[test]
fn test_tf_macro_domain_detection() {
    use dsmc::{Continuous, tf};

    // No type annotation: domain detected from the variable, T falls back to f64.
    let g = tf!("(s^2 + 10s + 10) / (s^2 + 20s + 100)");
    let _: &TransferFunction<f64, Continuous> = &g;
    assert_eq!(g.to_string(), "(s^2 + 10.0 * s + 10.0) / (s^2 + 20.0 * s + 100.0)");

    let h = tf!("0.5 / (z - 0.5)");
    let _: &TransferFunction<f64, Discrete> = &h;
    assert_eq!(h.to_string(), "0.5 / (z - 0.5)");

    // Token form is detected too.
    let k = tf!(0.5 / (z - 0.5));
    let _: &TransferFunction<f64, Discrete> = &k;

    // Placeholder names containing `s` / `z` are not mistaken for the variable.
    let zeta = 0.7;
    let wn = 10.0;
    let m = tf!("{wn}^2 / (s^2 + 2 * {zeta} * {wn} * s + {wn}^2)");
    let _: &TransferFunction<f64, Continuous> = &m;

    // Other float types via annotation.
    let k: TransferFunction<f32> = tf!("1 / (s + 1)");
    assert_eq!(k.to_string(), "1.0 / (s + 1.0)");
}

#[test]
fn test_parse_variable_errors() {
    use dsmc::TransferFunctionParseError;

    assert_eq!(
        "1 / (s + z)".parse::<TransferFunction<f64>>().unwrap_err(),
        TransferFunctionParseError::MixedVariables,
    );
    assert_eq!(
        "1 / (z + 1)".parse::<TransferFunction<f64>>().unwrap_err(),
        TransferFunctionParseError::WrongVariable { expected: 's', found: 'z', pos: 5 },
    );
    assert_eq!(
        "1 / (s + 1)".parse::<TransferFunction<f64, Discrete>>().unwrap_err(),
        TransferFunctionParseError::WrongVariable { expected: 'z', found: 's', pos: 5 },
    );
}

#[test]
fn test_tf_macro_with_format_args() {
    use dsmc::tf;

    let expected = TransferFunction::<f64>::continuous(&[100.0], &[1.0, 100.0]);

    // Inline captured variable
    let g = 100.0;
    let a = tf!("{g} / (s + {g})");
    assert_eq!(a.numerator, expected.numerator);
    assert_eq!(a.denominator, expected.denominator);

    // Positional arguments (expressions allowed)
    let b = tf!("{} / (s + {})", g, 2.0 * g / 2.0);
    assert_eq!(b.numerator, expected.numerator);
    assert_eq!(b.denominator, expected.denominator);

    // Negative values and values without a short decimal form survive the round trip.
    let p = -1.0 / 3.0;
    let c = tf!("1 / (z + {p})");
    let expected = TransferFunction::discrete(&[1.0], &[1.0, p]);
    assert_eq!(c.denominator, expected.denominator);
}

#[test]
fn test_transfer_function_scalar_gain() {
    use dsmc::tf;

    let g = tf!("1 / (s + 2)");

    let a = &g * 10.0;
    assert_eq!(a.to_string(), "10.0 / (s + 2.0)");

    let b = 10.0 * &g;
    assert_eq!(b.to_string(), "10.0 / (s + 2.0)");

    let c = g.clone() / 4.0;
    assert_eq!(c.to_string(), "0.25 / (s + 2.0)");

    let d = &g / 4.0;
    assert_eq!(d.to_string(), "0.25 / (s + 2.0)");

    // Compound assignment
    let mut e = g.clone();
    e *= 6.0;
    assert_eq!(e.to_string(), "6.0 / (s + 2.0)");
    e /= 3.0;
    assert_eq!(e.to_string(), "2.0 / (s + 2.0)");

    // Discrete and f32 work the same way.
    let h: TransferFunction<f32, Discrete> = tf!("1 / (z - 0.5)");
    assert_eq!((h * 2.0).to_string(), "2.0 / (z - 0.5)");

    // Left-hand gain without annotations on coefficients built from literals.
    let p = TransferFunction::continuous(&[1.0], &[1.0, 2.0]);
    assert_eq!((5.0 * p).to_string(), "5.0 / (s + 2.0)");

    // Combines with the other operators, e.g. a PI controller times a plant.
    let kp = 3.0;
    // 3(s + 1)/s * 1/(s + 2) = (3s + 3) / (s^2 + 2s)
    let loop_tf = &(kp * &tf!("(s + 1) / s")) * &g;
    assert_coeffs_close(&loop_tf.numerator, &[3.0, 3.0], 1e-9);
    assert_coeffs_close(&loop_tf.denominator, &[1.0, 2.0, 0.0], 1e-9);
}
