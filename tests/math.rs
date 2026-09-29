use dsmc::{TransferFunction, dka_method, vieta_formula};

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
    let tf = TransferFunction::<f64>::new(
        &[1.0, 10.0, 100.0],
        &[1.0, 20.0, 100.0],
    );

    let pz_map = tf.pz_map();
    println!("{:.2}", pz_map);
}

#[test]
fn test_transfer_function_parser() {
    use dsmc::tf;

    let expected = TransferFunction::<f64>::new(&[1.0, 10.0, 10.0], &[1.0, 20.0, 100.0]);

    let g: TransferFunction<f64> = tf!((s^2 + 10.0 * s + 10.0) / (s^2 + 20.0 * s + 100.0));
    assert_eq!(g.numerator, expected.numerator);
    assert_eq!(g.denominator, expected.denominator);

    let g = TransferFunction::<f64>::parse("(s^2 + 10s + 10) / (s + 10)^2").unwrap();
    assert_eq!(g.numerator, expected.numerator);
    assert_eq!(g.denominator, expected.denominator);

    // Pole-zero cancellation: (s+1)/((s+1)(s+2)) = 1/(s+2)
    let g: TransferFunction<f64> = "(s + 1) / ((s + 1)(s + 2)(s + 4))".parse().unwrap();
    println!("{:.2}", g);

    assert!(TransferFunction::<f64>::parse("1 / 0").is_err());
    assert!(TransferFunction::<f64>::parse("(s + 1").is_err());
    assert!(TransferFunction::<f64>::parse("s^0.5").is_err());
    assert!(TransferFunction::<f64>::parse("x + 1").is_err());
}

#[test]
fn test_transfer_function_display() {
    let g = TransferFunction::<f64>::parse("1 / (s + 2)").unwrap();
    assert_eq!(g.to_string(), "1.0 / (s + 2.0)");

    let g = TransferFunction::<f64>::parse("(s^2 - 3s) / (2s^2 + 20s + 100)").unwrap();
    println!("{}", g);
    println!("{:.2}", g);

    let g = TransferFunction::<f64>::parse("-s^2 + 1").unwrap();
    assert_eq!(g.to_string(), "-s^2 + 1.0");

    // Round trip
    let g = TransferFunction::<f64>::parse("(s^2 + 10s + 10) / (s^2 + 20s + 100)").unwrap();
    let h = TransferFunction::<f64>::parse(&g.to_string()).unwrap();
    assert_eq!(g.numerator, h.numerator);
    assert_eq!(g.denominator, h.denominator);
}
