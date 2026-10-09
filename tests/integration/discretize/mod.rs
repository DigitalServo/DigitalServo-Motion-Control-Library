//! Discretization of continuous-time models: every method compared (`methods`), and each one
//! against its closed form.
//!
//! The settings shared by the submodules are here: the sampling period, the test plants, and the
//! comparison of transfer functions.

mod matched_z;
mod methods;
mod tustin;
mod zoh;

use std::fmt::Debug;

use dsmc::{tf, Continuous, Discrete, DiscreteSystem, Polynomial, StateSpace, TransferFunction};

/// Sampling period \[s\].
const TS: f64 = 1e-3;

/// Test plants, one for each feature a discretization or a realization has to handle.
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
