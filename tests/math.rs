use mclib::{dka_method, vieta_formula};

#[test]
fn test_vieta_formula() {
    use num_complex::Complex;
    use mclib::Polynomial;

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
    use mclib::Polynomial;

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
fn test_roots_plot() {
    use num_complex::Complex;
    use mclib::Polynomial;
    use mclib::logger::DataStorage;

    let omega_c = -300.0;

    let roots: Vec<Complex<f64>> = vec![
        Complex::new(omega_c, 0.0),
        Complex::new(omega_c, 0.0),
        Complex::new(omega_c, 0.0),
    ];

    let coeffs = vieta_formula(&roots);

    let gains = coeffs.0[1..]
        .iter()
        .map(|x| x.re)
        .collect::<Vec<_>>();

    let iter = 20;
    for i in 0..=iter {
        let alpha = 0.8 + 0.4 / iter as f64 * i as f64;
        let gains_fluctuated = gains.iter().map(|x| x * alpha).collect::<Vec<_>>();

        let mut coeffs = vec![1.0];
        coeffs.extend_from_slice(&gains_fluctuated);
        let coeffs = coeffs
            .into_iter()
            .map(|x| Complex::new(x, 0.0))
            .collect::<Vec<_>>();

        let coeffs = Polynomial(coeffs);

        let roots = dka_method(&coeffs);
        println!("alpha: {:.2}, Roots: {:#.2?}", alpha, roots);

        let mut storage = DataStorage::new(format!("./out/roots_locus/roots_locus_alpha_{:.02}.csv", alpha), ',', false).unwrap();
        if let Some(roots) = roots {
            for root in roots {
                storage.add(&[root.re, root.im]).unwrap();
            }
            storage.close().unwrap();
        }
    }
}
