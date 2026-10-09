//! Zero-order hold: closed forms, step invariance.

use dsmc::discretize::Zoh;
use dsmc::{tf, Continuous, Polynomial, StateSpace, StateSpaceError, TransferFunction};
use nalgebra::dmatrix;

use super::{assert_same_tf, plants, step_response_z, TS};

/// `A_d = e^(A ts)`, `B_d = ∫_0^ts e^(Aτ) B dτ` in closed form for a double eigenvalue.
#[test]
fn state_space_formula() {
    let g = 10.0;
    let system = StateSpace::new(
        dmatrix![0.0, 1.0; -g * g, -2.0 * g],
        dmatrix![0.0; 1.0],
        dmatrix![g * g, 0.0],
        dmatrix![0.0],
    ).unwrap();
    let ssr_z = system.discretize(Zoh, TS).unwrap();

    // A has the double eigenvalue λ = -g, so e^{At} = e^{λt}(I + t(A - λI)) with
    // A - λI = [[g, 1], [-g^2, -g]]. Then
    //   A_d = e^{λT} [[1 + gT, T], [-g^2 T, 1 - gT]]
    //   B_d = ∫_0^T e^{Aτ} B dτ = [I1, I0 - g I1],  I0 = ∫ e^{λτ}, I1 = ∫ τ e^{λτ}
    let lambda: f64 = -g;
    let e = (lambda * TS).exp();
    let i0 = (e - 1.0) / lambda;
    let i1 = e * (TS / lambda - 1.0 / (lambda * lambda)) + 1.0 / (lambda * lambda);
    let expected_a = dmatrix![e * (1.0 + g * TS), e * TS; -g * g * TS * e, e * (1.0 - g * TS)];
    let expected_b = dmatrix![i1; i0 - g * i1];

    assert!((&ssr_z.a - &expected_a).abs().max() < 1e-12, "A_d = {} expected {}", ssr_z.a, expected_a);
    assert!((&ssr_z.b - &expected_b).abs().max() < 1e-15, "B_d = {} expected {}", ssr_z.b, expected_b);
    // C and D are unchanged by discretization.
    assert_eq!(ssr_z.c, system.c);
    assert_eq!(ssr_z.d, system.d);
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

/// `a / (s + a)` -> `(1 - p) / (z - p)`, `p = e^(-a ts)`.
#[test]
fn first_order() {
    let p = (-100.0 * TS).exp();
    let gz = tf!("100 / (s + 100)").discretize(Zoh, TS).unwrap();
    let expected = TransferFunction::discrete(&[1.0 - p], &[1.0, -p]);
    assert_same_tf(&gz, &expected, 1e-15, "ZOH");
}

#[test]
fn static_gain_and_improper() {
    let gain = TransferFunction::<f64, Continuous>::from_polynomials(Polynomial(vec![3.0]), Polynomial(vec![2.0]));
    let gz = gain.discretize(Zoh, TS).unwrap();
    assert_eq!((gz.numerator.0.clone(), gz.denominator.0.clone()), (vec![3.0], vec![2.0]));

    let improper = TransferFunction::<f64, Continuous>::from_polynomials(Polynomial(vec![1.0, 1.0, 1.0]), Polynomial(vec![1.0, 2.0]));
    assert_eq!(improper.discretize(Zoh, TS).unwrap_err(), StateSpaceError::Improper { numerator: 2, denominator: 1 });
}
