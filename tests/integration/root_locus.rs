use dsmc::{dka_method, vieta_formula};
use num_complex::Complex;

/// |p(r)| relative to the size of the terms, i.e. how well `r` satisfies `p(r) = 0`.
fn relative_residual(coeffs: &[Complex<f64>], r: Complex<f64>) -> f64 {
    let (mut value, mut scale) = (Complex::new(0.0, 0.0), 0.0);
    for (k, c) in coeffs.iter().enumerate() {
        let term = c * r.powi((coeffs.len() - 1 - k) as i32);
        value += term;
        scale += term.norm();
    }
    value.norm() / scale
}

#[test]
fn test_roots_plot() {

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

        // Every root satisfies the cubic, and complex roots come in conjugate pairs.
        let found = roots.as_ref().expect("dka_method failed");
        assert_eq!(found.len(), 3, "alpha = {alpha}");
        for r in found {
            assert!(relative_residual(&coeffs.0, *r) < 1e-9, "alpha = {alpha}: {r} is not a root");
            assert!(found.iter().any(|q| (q - r.conj()).norm() < 1e-6 * r.norm()), "alpha = {alpha}: no conjugate for {r}");
        }
        // alpha = 1 reproduces the triple root; its accuracy is only ~eps^(1/3).
        if (alpha - 1.0).abs() < 1e-12 {
            for r in found {
                assert!((r - Complex::new(omega_c, 0.0)).norm() < 1e-2, "triple root: {r}");
            }
        }

        let mut storage = DataStorage::new(format!("./out/roots_locus/roots_locus_alpha_{:.02}.csv", alpha)).unwrap();
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

        // Analytic roots: -ζω ± ω√(ζ² - 1) (complex for ζ < 1). The double root at ζ = 1 is only
        // found to ~sqrt(eps) accuracy, hence the loose tolerance.
        let disc = Complex::new(zeta * zeta - 1.0, 0.0).sqrt() * omega_c;
        let expected = [Complex::new(-zeta * omega_c, 0.0) + disc, Complex::new(-zeta * omega_c, 0.0) - disc];
        let found = roots.as_ref().expect("dka_method failed");
        assert_eq!(found.len(), 2, "zeta = {zeta}");
        for e in expected {
            assert!(found.iter().any(|r| (r - e).norm() < 1e-3), "zeta = {zeta}: {e} not in {found:?}");
        }

        let mut storage = DataStorage::new(format!("./out/roots_locus_2order/roots_locus_zeta_{:.01}.csv", zeta)).unwrap();
        if let Some(roots) = roots {
            for root in roots {
                storage.add(&[root.re, root.im]).unwrap();
            }
            storage.close().unwrap();
        }
    }
}
