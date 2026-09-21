use dsmc::{dka_method, vieta_formula};

#[test]
fn test_roots_plot() {
    use num_complex::Complex;
    use dsmc::Polynomial;
    use dsmc::logger::DataStorage;

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



#[test]
fn test_roots_plot_2order() {
    use num_complex::Complex;
    use dsmc::Polynomial;
    use dsmc::logger::DataStorage;

    let omega_c = 300.0;

    let iter = 20;
    for i in 0..=iter {
        let zeta = 0.0 + 2.0 / iter as f64 * i as f64;

        let coeffs = vec![1.0, 2.0 * zeta * omega_c, omega_c * omega_c];

        let coeffs = coeffs
            .into_iter()
            .map(|x| Complex::new(x, 0.0))
            .collect::<Vec<_>>();

        let coeffs = Polynomial(coeffs);

        let roots = dka_method(&coeffs);
        println!("zeta: {:.2}, Roots: {:#.2?}", zeta, roots);

        let mut storage = DataStorage::new(format!("./out/roots_locus_2order/roots_locus_zeta_{:.01}.csv", zeta), ',', false).unwrap();
        if let Some(roots) = roots {
            for root in roots {
                storage.add(&[root.re, root.im]).unwrap();
            }
            storage.close().unwrap();
        }
    }
}
