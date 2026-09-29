use dsmc::{Discrete, TransferFunction, dka_method, vieta_formula};

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

    let coeffs = vieta_formula(&roots);

    println!("Roots: {:#.2?}", roots);
    println!("Coefficients: {:#.2?}", coeffs);

    let coeffs = coeffs.0
        .into_iter()
        .map(|x| Complex{re: 1.2, im: 3.0} * x)
        .collect::<Vec<_>>();
    let coeffs = Polynomial(coeffs);

    let roots = dka_method(&coeffs);
    println!("Roots: {:#.2?}", roots);
}

#[test]
fn test_search_roots() {
    use num_complex::Complex;
    use dsmc::Polynomial;

    let coeffs = Polynomial(vec![
        Complex::<f64>::new(1.0, 0.0),
        Complex::<f64>::new(20.0, 0.0),
        Complex::<f64>::new(100.0, 0.0),
    ]);

    println!("Coefficients: {:#.2?}", coeffs);

    let coeffs = coeffs.0
        .into_iter()
        .map(|x| Complex{re: 1.2, im: 3.0} * x)
        .collect::<Vec<_>>();
    let coeffs = Polynomial(coeffs);

    let roots = dka_method(&coeffs);
    println!("Roots: {:#.2?}", roots);
}


#[test]
fn test_pz_map() {
    let tf = TransferFunction::<f64>::continuous(
        &[1.0, 10.0, 100.0],
        &[1.0, 20.0, 100.0],
    );

    let pz_map = tf.pz_map();
    println!("{:.2}", pz_map);
}

#[test]
fn test_transfer_function_parser() {
    use dsmc::tf;

    let expected = TransferFunction::<f64>::continuous(&[1.0, 10.0, 10.0], &[1.0, 20.0, 100.0]);

    let g: TransferFunction<f64> = tf!((s^2 + 10.0 * s + 10.0) / (s^2 + 20.0 * s + 100.0));
    assert_eq!(g.numerator, expected.numerator);
    assert_eq!(g.denominator, expected.denominator);

    let g = tf!("(s^2 + 10s + 10) / (s + 10)^2", 's');
    assert_eq!(g.numerator, expected.numerator);
    assert_eq!(g.denominator, expected.denominator);

    // Pole-zero cancellation: (s+1)/((s+1)(s+2)) = 1/(s+2)
    let g = tf!("(s + 1) / ((s + 1)(s + 2)(s + 4))", 's');
    println!("{:.2}", g);

    // Errors are only observable through `FromStr`, since `tf!` panics.
    assert!("1 / 0".parse::<TransferFunction<f64>>().is_err());
    assert!("(s + 1".parse::<TransferFunction<f64>>().is_err());
    assert!("s^0.5".parse::<TransferFunction<f64>>().is_err());
    assert!("x + 1".parse::<TransferFunction<f64>>().is_err());
}

#[test]
fn test_transfer_function_display() {
    use dsmc::tf;

    let g = tf!("1 / (s + 2)", 's');
    assert_eq!(g.to_string(), "1.0 / (s + 2.0)");

    let g = tf!("(s^2 - 3s) / (2s^2 + 20s + 100)", 's');
    println!("{}", g);
    println!("{:.2}", g);

    let g = tf!("-s^2 + 1", 's');
    assert_eq!(g.to_string(), "-s^2 + 1.0");

    // Round trip
    let g = tf!("(s^2 + 10s + 10) / (s^2 + 20s + 100)", 's');
    let h: TransferFunction<f64> = g.to_string().parse().unwrap();
    assert_eq!(g.numerator, h.numerator);
    assert_eq!(g.denominator, h.denominator);
}

#[test]
fn test_discrete_transfer_function() {
    use dsmc::tf;

    // Annotation needed here: `T` falls back to f64 only at the end of type checking,
    // too late for the `.abs()` call on the pole below.
    let g: TransferFunction<f64, Discrete> = tf!("0.5 / (z - 0.5)", 'z');
    assert_eq!(g.to_string(), "0.5 / (z - 0.5)");

    let pz = g.pz_map();
    assert!((pz.poles[0].re - 0.5).abs() < 1e-9);

    // `s` is not a valid variable in the z-domain, and vice versa.
    assert!("1 / (s + 1)".parse::<TransferFunction<f64, Discrete>>().is_err());
    assert!("1 / (z + 1)".parse::<TransferFunction<f64>>().is_err());
}

#[test]
fn test_tf_macro_with_variable() {
    use dsmc::{Continuous, tf};

    // No type annotation: domain from the variable, T falls back to f64.
    let g = tf!("(s^2 + 10s + 10) / (s^2 + 20s + 100)", 's');
    let _: &TransferFunction<f64, Continuous> = &g;
    assert_eq!(g.to_string(), "(s^2 + 10.0 * s + 10.0) / (s^2 + 20.0 * s + 100.0)");

    let h = tf!("0.5 / (z - 0.5)", 'z');
    let _: &TransferFunction<f64, Discrete> = &h;
    assert_eq!(h.to_string(), "0.5 / (z - 0.5)");

    // Other float types via annotation.
    let k: TransferFunction<f32> = tf!("1 / (s + 1)", 's');
    assert_eq!(k.to_string(), "1.0 / (s + 1.0)");
}

#[test]
fn test_tf_macro_with_format_args() {
    use dsmc::tf;

    let expected = TransferFunction::<f64>::continuous(&[100.0], &[1.0, 100.0]);

    // Inline captured variable
    let g = 100.0;
    let a = tf!("{g} / (s + {g})", 's');
    assert_eq!(a.numerator, expected.numerator);
    assert_eq!(a.denominator, expected.denominator);

    // Positional arguments (expressions allowed)
    let b = tf!("{} / (s + {})", 's', g, 2.0 * g / 2.0);
    assert_eq!(b.numerator, expected.numerator);
    assert_eq!(b.denominator, expected.denominator);

    // Negative values and values without a short decimal form survive the round trip.
    let p = -1.0 / 3.0;
    let c = tf!("1 / (z + {p})", 'z');
    let expected = TransferFunction::discrete(&[1.0], &[1.0, p]);
    assert_eq!(c.denominator, expected.denominator);
}
