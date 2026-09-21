use dsmc::{dka_method, vieta_formula};

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
