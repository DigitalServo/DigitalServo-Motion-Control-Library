use dsmc::{
    DiscreteSystem, StateSpace, TransferFunction, discretize::Zoh, logger::DataStorage
};
use nalgebra::dmatrix;

#[test]
fn test_zoh_ssr() {

    let ts = 1.0e-4;

    let g = 10.0;
    let system = StateSpace::new(
        dmatrix![0.0, 1.0; -g * g, -2.0 * g],
        dmatrix![0.0; 1.0],
        dmatrix![g * g, 0.0],
        dmatrix![0.0],
    ).unwrap();
    let ssr_z = system.discretize(Zoh, ts).unwrap();

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
fn test_zoh_system_ssr() {

    let ts = 1.0e-4;
    let mut storage = DataStorage::new("./out/zoh_ssr_out.csv", ',', false).unwrap();

    let g = 10.0;
    let system = StateSpace::new(
        dmatrix![0.0, 1.0; -g * g, -2.0 * g],
        dmatrix![0.0; 1.0],
        dmatrix![g * g, 0.0],
        dmatrix![0.0],
    ).unwrap();
    let mut system = DiscreteSystem::from(system.discretize(Zoh, ts).unwrap());

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
fn test_zoh_system_tf() {

    let ts = 1.0e-4;
    let mut storage = DataStorage::new("./out/zoh_tf_out.csv", ',', false).unwrap();

    let g = 10.0;
    let system = TransferFunction::continuous(&[g * g], &[1.0, 2.0 * g, g * g]);
    let mut system = DiscreteSystem::try_from(&system.discretize(Zoh, ts).unwrap()).unwrap();

    let mut t = 0.0;
    for _ in 0..20000 {
        let x = if t < 0.2 { 0.0 } else { 1.0 };
        let y = system.update(x);
        storage.add(&[t, x, y]).unwrap();

        t += ts;
    }

    storage.close().unwrap();
}

mod transfer_function {
    use dsmc::discretize::Zoh;
    use dsmc::{tf, Continuous, Polynomial, DiscreteSystem, StateSpace, StateSpaceError, TransferFunction};

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

    /// Step response of G(z) at t = k ts (`DiscreteSystem`: the difference equation in state form).
    fn step_response_z(g: &TransferFunction<f64, dsmc::Discrete>, samples: usize) -> Vec<f64> {
        let mut system = DiscreteSystem::try_from(g).unwrap();
        (0..samples).map(|_| system.update(1.0)).collect()
    }

    /// Step invariance: the step responses agree with the continuous one at the sampling instants.
    #[test]
    fn step_invariant() {
        for g in plants() {
            let gz = g.discretize(Zoh, TS).unwrap();
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
        let gz = tf!("100 / (s + 100)").discretize(Zoh, TS).unwrap();
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

    /// The discretized canonical realizations (standard and normalized) work for any proper G(s),
    /// including a zero at the origin (N(0) = 0), and simulate the same step response as the
    /// discretized transfer function.
    #[test]
    fn realizations_simulate_like_tf() {
        for g in plants() {
            // `update` returns y[k] for u[k], like the difference equation of G(z).
            let expected = step_response_z(&g.discretize(Zoh, TS).unwrap(), 300);
            let realizations = [StateSpace::try_from(&g).unwrap(), StateSpace::normalized_controllable_canonical(&g).unwrap()];
            for mut system in realizations.map(|ssr| DiscreteSystem::from(ssr.discretize(Zoh, TS).unwrap())) {
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
        let gz = gain.discretize(Zoh, TS).unwrap();
        assert_eq!((gz.numerator.0.clone(), gz.denominator.0.clone()), (vec![3.0], vec![2.0]));

        let improper = TransferFunction::<f64, Continuous>::from_polynomials(Polynomial(vec![1.0, 1.0, 1.0]), Polynomial(vec![1.0, 2.0]));
        assert_eq!(improper.discretize(Zoh, TS).unwrap_err(), StateSpaceError::Improper { numerator: 2, denominator: 1 });
    }

    /// A transfer function is simulated as its difference equation: the leading denominator
    /// coefficient need not be 1, leading zero coefficients are skipped, a static gain works,
    /// and `reset` goes back to rest. The model itself is left unchanged.
    #[test]
    fn discrete_system_of_transfer_function() {
        use dsmc::Discrete;
        // 2 y[k] - y[k-1] = u[k-1]  (G = 1 / (2z - 1), written with a leading zero in the numerator)
        let g = TransferFunction::<f64, Discrete>::from_polynomials(Polynomial(vec![0.0, 1.0]), Polynomial(vec![2.0, -1.0]));
        let mut system = DiscreteSystem::try_from(&g).unwrap();
        let y: Vec<f64> = (0..4).map(|_| system.update(1.0)).collect();
        for (yk, e) in y.iter().zip([0.0, 0.5, 0.75, 0.875]) {
            assert!((yk - e).abs() < 1e-15, "{y:?}");
        }
        system.reset();
        assert_eq!(system.update(1.0), 0.0);
        assert_eq!(system.model().order.system, 1);

        // Static gain
        let k = TransferFunction::<f64, Discrete>::from_polynomials(Polynomial(vec![3.0]), Polynomial(vec![2.0]));
        assert_eq!(DiscreteSystem::try_from(&k).unwrap().update(2.0), 3.0);

        // Improper (not causal)
        let improper = TransferFunction::<f64, Discrete>::from_polynomials(Polynomial(vec![1.0, 0.0]), Polynomial(vec![1.0]));
        assert!(matches!(DiscreteSystem::try_from(&improper), Err(StateSpaceError::Improper { .. })));

    }

    /// `Siso` and `Mimo` run the same model; converting between them keeps the state. `Siso` needs
    /// one input and one output, `Mimo` checks the length of the input.
    #[test]
    fn siso_and_mimo() {
        use dsmc::{Discrete, Mimo, Siso};
        use nalgebra::dmatrix;

        let gz = tf!("1000 / (s^2 + 20s + 1000)").discretize(Zoh, TS).unwrap();
        let mut siso: DiscreteSystem<f64, Siso> = DiscreteSystem::try_from(&gz).unwrap();
        let mut mimo: DiscreteSystem<f64, Mimo> = DiscreteSystem::from(StateSpace::try_from(&gz).unwrap());
        for k in 0..50 {
            let u = (k as f64 * 0.3).sin();
            let (y_siso, y_mimo) = (siso.update(u), mimo.update(&[u]).unwrap()[0]);
            assert!((y_siso - y_mimo).abs() < 1e-15, "at k = {k}: {y_siso} vs {y_mimo}");
            assert_eq!(siso.output[0], y_siso);
        }

        // Switching keeps the state: both continue identically.
        let mut back: DiscreteSystem<f64, Siso> = DiscreteSystem::try_from(mimo.clone()).unwrap();
        let mut again: DiscreteSystem<f64, Mimo> = DiscreteSystem::from(siso.clone());
        for _ in 0..10 {
            assert_eq!(back.update(1.0), siso.update(1.0));
            assert_eq!(again.update(&[1.0]).unwrap(), mimo.update(&[1.0]).unwrap());
        }

        assert!(matches!(mimo.update(&[1.0, 2.0]), Err(StateSpaceError::InputVector { .. })));
        let two_inputs = StateSpace::<f64, Discrete>::new(dmatrix![0.5], dmatrix![1.0, 1.0], dmatrix![1.0], dmatrix![0.0, 0.0]).unwrap();
        assert_eq!(
            DiscreteSystem::<f64, Siso>::try_from(&two_inputs).unwrap_err(),
            StateSpaceError::NotSiso { inputs: 2, outputs: 1 }
        );
    }
}
