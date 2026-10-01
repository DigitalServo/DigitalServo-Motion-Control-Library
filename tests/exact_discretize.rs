use dsmc::{
    StateSpace, TransferFunction, discretize::exact_discretize::{DiscretizedSystem, discretize_ssr}, logger::DataStorage
};
use nalgebra::dmatrix;

#[test]
fn test_exact_discretize_ssr() {

    let ts = 1.0e-4;

    let g = 10.0;
    let system = StateSpace::new(
        dmatrix![0.0, 1.0; -g * g, -2.0 * g],
        dmatrix![0.0; 1.0],
        dmatrix![g * g, 0.0],
        dmatrix![0.0],
    ).unwrap();
    let ssr_z = discretize_ssr(&system, ts).unwrap();

    // A has the double eigenvalue λ = -g, so e^{At} = e^{λt}(I + t(A - λI)) with
    // A - λI = [[g, 1], [-g^2, -g]]. Then
    //   A_d = e^{λT} [[1 + gT, T], [-g^2 T, 1 - gT]]
    //   B_d = ∫_0^T e^{Aτ} B dτ = [I1, I0 - g I1],  I0 = ∫ e^{λτ}, I1 = ∫ τ e^{λτ}
    let lambda: f64 = -g;
    let e = (lambda * ts).exp();
    let i0 = (e - 1.0) / lambda;
    let i1 = e * (ts / lambda - 1.0 / (lambda * lambda)) + 1.0 / (lambda * lambda);
    let expected_a = dmatrix![e * (1.0 + g * ts), e * ts; -g * g * ts * e, e * (1.0 - g * ts)];
    let expected_b = dmatrix![i1; i0 - g * i1];

    assert!((&ssr_z.a - &expected_a).abs().max() < 1e-12, "A_d = {} expected {}", ssr_z.a, expected_a);
    assert!((&ssr_z.b - &expected_b).abs().max() < 1e-15, "B_d = {} expected {}", ssr_z.b, expected_b);
    // C and D are unchanged by discretization.
    assert_eq!(ssr_z.c, system.c);
    assert_eq!(ssr_z.d, system.d);
}

#[test]
fn test_exact_discretize_system_ssr() {

    let ts = 1.0e-4;
    let mut storage = DataStorage::new("./out/exact_discretized_ssr_out.csv", ',', false).unwrap();

    let g = 10.0;
    let system = StateSpace::new(
        dmatrix![0.0, 1.0; -g * g, -2.0 * g],
        dmatrix![0.0; 1.0],
        dmatrix![g * g, 0.0],
        dmatrix![0.0],
    ).unwrap();
    let mut system = DiscretizedSystem::from_ssr(system, ts).unwrap();

    let mut t = 0.0;
    for _ in 0..20000 {
        let x = if t < 0.2 { 0.0 } else { 1.0 };
        let y = system.update(&[x]).unwrap();
        storage.add(&[t, x, y[0]]).unwrap();

        t += ts;
    }

    storage.close().unwrap();
}


#[test]
fn test_exact_discretize_system_tf() {

    let ts = 1.0e-4;
    let mut storage = DataStorage::new("./out/exact_discretized_tf_out.csv", ',', false).unwrap();

    let g = 10.0;
    let system = TransferFunction::continuous(&[g * g], &[1.0, 2.0 * g, g * g]);
    let mut system = DiscretizedSystem::from_tf(system, ts).unwrap();

    let mut t = 0.0;
    for _ in 0..20000 {
        let x = if t < 0.2 { 0.0 } else { 1.0 };
        let y = system.update(&[x]).unwrap();
        storage.add(&[t, x, y[0]]).unwrap();

        t += ts;
    }

    storage.close().unwrap();
}

mod transfer_function {
    use dsmc::discretize::{bilinear_transform, exact_discretize::discretize};
    use dsmc::{tf, Continuous, Polynomial, StateSpace, StateSpaceError, TransferFunction};

    const TS: f64 = 1e-3;

    fn plants() -> Vec<TransferFunction<f64, Continuous>> {
        vec![
            tf!("100 / (s + 100)"),
            tf!("1000 / (s^2 + 20s + 1000)"),
            tf!("(1 - 0.01s) / ((s + 20)^2)"),     // nonminimum phase, repeated pole
            tf!("(s + 50) / ((s + 10) (s^2 + 40s + 10000))"),
            tf!("s / ((s + 10) (s + 30))"),          // zero at the origin
            tf!("(2s + 30) / (s + 10)"),             // biproper (direct term 2)
            tf!("1 / (s (0.02s + 1))"),              // integrator
        ]
    }

    /// Step response of G(z) at t = k ts by the difference equation.
    fn step_response_z(g: &TransferFunction<f64, dsmc::Discrete>, samples: usize) -> Vec<f64> {
        let mut system = bilinear_transform::DiscretizedSystem::from_tf_z(g);
        (0..samples).map(|_| system.update(1.0)).collect()
    }

    /// Step invariance: the step responses agree with the continuous one at the sampling instants.
    #[test]
    fn step_invariant() {
        for g in plants() {
            let gz = discretize(&g, TS).unwrap();
            // y(t) = L^-1[G(s) / s] (exact, by partial fractions), sampled at t = k ts
            let step = TransferFunction::<f64, Continuous>::from_polynomials(
                g.numerator.clone(),
                &g.denominator * &Polynomial(vec![1.0, 0.0]),
            )
            .inverse_laplace();
            // With a direct term, y[0] = d (the value just after the step).
            for (k, yk) in step_response_z(&gz, 500).into_iter().enumerate() {
                let t = k as f64 * TS;
                let expected = if k == 0 { step(1e-12) } else { step(t) };
                assert!((yk - expected).abs() < 1e-8 * expected.abs().max(1.0), "{g:?} at k = {k}: {yk} vs {expected}");
            }
        }
    }

    #[test]
    fn first_order() {
        // a / (s + a) -> (1 - p) / (z - p), p = e^(-a ts)
        let p = (-100.0 * TS).exp();
        let gz = discretize(tf!("100 / (s + 100)"), TS).unwrap();
        let k = gz.denominator[0];
        let numer: Vec<f64> = gz.numerator.iter().map(|c| c / k).skip_while(|c| c.abs() < 1e-15).collect();
        let denom: Vec<f64> = gz.denominator.iter().map(|c| c / k).collect();
        assert!((numer[0] - (1.0 - p)).abs() < 1e-15 && numer.len() == 1, "{numer:?}");
        assert!((denom[1] + p).abs() < 1e-15, "{denom:?}");
    }

    /// The canonical realization and back reproduce the transfer function.
    #[test]
    fn realization_round_trip() {
        for g in plants() {
            let back = StateSpace::controllable_canonical(&g).unwrap().transfer_function().unwrap();
            let lead = g.denominator.iter().copied().find(|c| *c != 0.0).unwrap();
            let expected_n: Vec<f64> = g.numerator.iter().map(|c| c / lead).collect();
            let expected_d: Vec<f64> = g.denominator.iter().map(|c| c / lead).collect();
            let trim = |v: &[f64]| v.iter().copied().skip_while(|c| *c == 0.0).collect::<Vec<_>>();
            let (n, d) = (trim(&back.numerator), trim(&back.denominator));
            assert_eq!((n.len(), d.len()), (trim(&expected_n).len(), trim(&expected_d).len()), "{g:?} vs {back:?}");
            for (x, e) in n.iter().chain(&d).zip(trim(&expected_n).iter().chain(&trim(&expected_d))) {
                assert!((x - e).abs() < 1e-9 * e.abs().max(1.0), "{g:?} vs {back:?}");
            }
        }
    }

    /// `x = u / D(s)`, `y = N(s) x` with N and D divided by N(0).
    #[test]
    fn normalized_realization() {
        for g in plants().into_iter().filter(|g| g.numerator.last() != Some(&0.0)) {
            let ssr = StateSpace::normalized_controllable_canonical(&g).unwrap();
            // Same transfer function as the standard canonical form
            let a = ssr.transfer_function().unwrap();
            let b = StateSpace::controllable_canonical(&g).unwrap().transfer_function().unwrap();
            for (x, y) in a.numerator.iter().chain(a.denominator.iter()).zip(b.numerator.iter().chain(b.denominator.iter())) {
                assert!((x - y).abs() < 1e-9 * y.abs().max(1.0), "{a:?} vs {b:?}");
            }
            // Normalized numerator at s = 0 is 1: y = N(s) x = C x + d u with D'(s) x = N'(0) u, so
            // N(0) = C_0 + d a'_0 / N'(0) (a'_0 = -A[n-1, 0], N'(0) = B[n-1])
            let n = ssr.order.system;
            let n_at_zero = ssr.c[(0, 0)] + ssr.d[(0, 0)] * (-ssr.a[(n - 1, 0)]) / ssr.b[(n - 1, 0)];
            assert!((n_at_zero - 1.0).abs() < 1e-12, "{g:?}: C = {}, B = {}", ssr.c, ssr.b);
        }

        // Without zeros, y = x1 exactly.
        let ssr = StateSpace::normalized_controllable_canonical(&tf!("1000 / (s^2 + 20s + 1000)")).unwrap();
        assert_eq!(ssr.c, nalgebra::dmatrix![1.0, 0.0]);

        // N(0) = 0: N = s^k N_1 is normalized by N_1(0), so that y ≈ ξ^(k) = x_(k+1) at low frequency.
        // 5s / ((s + 10)(s + 30)) -> y = ξ' exactly (C = [0, 1]), D(s) ξ = 5 u
        let ssr = StateSpace::normalized_controllable_canonical(&tf!("5s / ((s + 10) (s + 30))")).unwrap();
        assert_eq!(ssr.c, nalgebra::dmatrix![0.0, 1.0]);
        assert_eq!(ssr.b, nalgebra::dmatrix![0.0; 5.0]);
        // With zeros besides the origin: the lowest-order coefficient of the normalized numerator is 1.
        // (3s^2 + 6s) / (s^3 + 4s^2 + 5s + 2) = 3 s (s + 2) / ...: N_1(0) = 6
        let ssr = StateSpace::normalized_controllable_canonical(&TransferFunction::from_polynomials(
            Polynomial(vec![3.0, 6.0, 0.0]),
            Polynomial(vec![1.0, 4.0, 5.0, 2.0]),
        ))
        .unwrap();
        assert_eq!(ssr.c, nalgebra::dmatrix![0.0, 1.0, 0.5]);
    }

    /// `from_tf` and `from_tf_normalized` work for any proper G(s), including a zero at the origin
    /// (N(0) = 0), and simulate the same step response as `discretize`.
    #[test]
    fn from_tf_is_general() {
        use dsmc::discretize::exact_discretize::DiscretizedSystem;
        for g in plants() {
            // `update` returns y[k] for u[k], like the difference equation of G(z).
            let expected = step_response_z(&discretize(&g, TS).unwrap(), 300);
            for mut system in [DiscretizedSystem::from_tf(&g, TS).unwrap(), DiscretizedSystem::from_tf_normalized(&g, TS).unwrap()] {
                for (k, &e) in expected.iter().enumerate() {
                    let y = system.update(&[1.0]).unwrap()[0];
                    assert!((y - e).abs() < 1e-9 * e.abs().max(1.0), "{g:?} at k = {k}: {y} vs {e}");
                }
            }
        }
    }

    #[test]
    fn static_gain_and_improper() {
        let gain = TransferFunction::<f64, Continuous>::from_polynomials(Polynomial(vec![3.0]), Polynomial(vec![2.0]));
        let gz = discretize(&gain, TS).unwrap();
        assert_eq!((gz.numerator.0.clone(), gz.denominator.0.clone()), (vec![3.0], vec![2.0]));

        let improper = TransferFunction::<f64, Continuous>::from_polynomials(Polynomial(vec![1.0, 1.0, 1.0]), Polynomial(vec![1.0, 2.0]));
        assert_eq!(discretize(&improper, TS).unwrap_err(), StateSpaceError::Improper { numerator: 2, denominator: 1 });
    }
}
