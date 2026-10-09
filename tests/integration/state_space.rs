//! Canonical realizations of transfer functions and their simulation by `DiscreteSystem`.

use std::fmt::Debug;

use dsmc::discretize::Zoh;
use dsmc::{tf, Continuous, Discrete, DiscreteSystem, Polynomial, StateSpace, StateSpaceError, TransferFunction};

/// Sampling period \[s\].
const TS: f64 = 1e-3;

/// Test plants, one for each feature a realization has to handle.
fn plants() -> Vec<TransferFunction<f64, Continuous>> {
    vec![
        tf!("100 / (s + 100)"),
        tf!("1000 / (s^2 + 20s + 1000)"),
        tf!("(1 - 0.01s) / ((s + 20)^2)"),                  // nonminimum phase, repeated pole
        tf!("(s + 50) / ((s + 10) (s^2 + 40s + 10000))"),
        tf!("(2s + 30) / (s + 10)"),                          // biproper (direct term 2)
        tf!("s / ((s + 10) (s + 30))"),                       // zero at the origin
        tf!("1 / (s (0.02s + 1))"),                           // integrator
        tf!("(s + 5) / (s^2 (s + 100))"),                     // double integrator
        tf!("10 / (s + 0.001)"),                              // slow pole, z = 1 - 1e-6 (not an integrator)
    ]
}

/// Numerator and denominator coefficients divided by the leading denominator coefficient, with
/// the leading coefficients of each polynomial below `1e-14` times its largest one left out
/// (zero up to the rounding).
fn normalized<D>(g: &TransferFunction<f64, D>) -> (Vec<f64>, Vec<f64>) {
    let trim = |p: &Polynomial<f64>| {
        let largest = p.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        p.iter().copied().skip_while(|c| c.abs() <= 1e-14 * largest).collect::<Vec<f64>>()
    };
    let (n, d) = (trim(&g.numerator), trim(&g.denominator));
    (n.iter().map(|c| c / d[0]).collect(), d.iter().map(|c| c / d[0]).collect())
}

/// `|actual - expected| <= tol max(|expected|, 1)` coefficient by coefficient, with the same length.
fn assert_coeffs_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: {actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tol * e.abs().max(1.0), "{what}: {actual:?} vs {expected:?}");
    }
}

/// `a` and `b` have the same `normalized` coefficients within `tol` (`assert_coeffs_close`).
fn assert_same_tf<D>(a: &TransferFunction<f64, D>, b: &TransferFunction<f64, D>, tol: f64, what: &str)
where
    TransferFunction<f64, D>: Debug,
{
    let ((an, ad), (bn, bd)) = (normalized(a), normalized(b));
    assert_coeffs_close(&an, &bn, tol, &format!("{what}: numerator of {a:?} vs {b:?}"));
    assert_coeffs_close(&ad, &bd, tol, &format!("{what}: denominator of {a:?} vs {b:?}"));
}

/// Step response of G(z) at t = k ts (`DiscreteSystem`: the difference equation in state form).
fn step_response_z(g: &TransferFunction<f64, Discrete>, samples: usize) -> Vec<f64> {
    let mut system = DiscreteSystem::try_from(g).unwrap();
    (0..samples).map(|_| system.update(1.0)).collect()
}

/// Step response (first output) of a single-input discrete state-space model.
fn step_response_ss(model: StateSpace<f64, Discrete>, samples: usize) -> Vec<f64> {
    let mut system = DiscreteSystem::from(model);
    (0..samples).map(|_| system.update(&[1.0]).unwrap()[0]).collect()
}

/// The canonical realization and back reproduce the transfer function.
#[test]
fn realization_round_trip() {
    for g in plants() {
        let back = StateSpace::controllable_canonical(&g).unwrap().transfer_function().unwrap();
        assert_same_tf(&back, &g, 1e-9, "realization");
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
        assert_same_tf(&a, &b, 1e-9, "normalized realization");
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
        for ssr in realizations {
            let y = step_response_ss(ssr.discretize(Zoh, TS).unwrap(), 300);
            for (k, (yk, e)) in y.iter().zip(&expected).enumerate() {
                assert!((yk - e).abs() < 1e-9 * e.abs().max(1.0), "{g:?} at k = {k}: {yk} vs {e}");
            }
        }
    }
}

/// A transfer function is simulated as its difference equation: the leading denominator
/// coefficient need not be 1, leading zero coefficients are skipped, a static gain works,
/// and `reset` goes back to rest. The model itself is left unchanged.
#[test]
fn discrete_system_of_transfer_function() {
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
    use dsmc::{Mimo, Siso};
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
