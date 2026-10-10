//! Pure integrators `b_0 / s^q` (`n = 0`, `A = 1`) by `identify_with_prefilter` and `search`.

use std::f64::consts::PI;

use dsmc::system_identification::iv::srivc::{self, identify, identify_with_prefilter, Initialization, Outcome, Prefilter, SearchOptions, SrivcError, SrivcOptions, Structure};
use dsmc::system_identification::validation::{Check, CoherenceCheck};

use super::*;

/// Prefilter cutoff \[rad/s\], below the excited band (10 Hz).
const CUTOFF: f64 = 2.0 * PI * 3.0;

/// Single inertia `J = 0.01`, `1 / (J s^q)` (`q = 1`: torque -> velocity, `q = 2`: torque ->
/// position): `(n, m, q) = (0, 0, q)` gives `b_0 = 1 / J`, the model `b_0 / s^q`. The regression
/// `z = b_0 w` holds no output, so the estimate is the least squares one whatever the
/// initialization: one iteration from `StateVariableFilter` (which is that estimate), two from a
/// `Model` (the first moves to it).
#[test]
fn test_srivc_pure_integrator() {
    let j = 0.01;
    let options = SrivcOptions { evaluated_from: PERIOD, ..SrivcOptions::default() };
    let constant = TransferFunction::continuous(&[50.0], &[1.0]);
    for q in [1, 2] {
        let mut denominator = vec![j];
        denominator.resize(q + 1, 0.0);
        let plant = TransferFunctionWithDelay::from(TransferFunction::continuous(&[1.0], &denominator));
        let (u, y) = experiment(&plant, 3 * PERIOD, 0x1234_5678_9abc, 0.05, 0.0);
        let prefilter = Prefilter::new(q, CUTOFF).unwrap();
        let mut estimates = Vec::new();
        for (init, iterations) in [(Initialization::StateVariableFilter(2.0 * PI * 100.0), 1), (Initialization::Model(constant.clone()), 2)] {
            let result = identify_with_prefilter(&u, &y, TS, 0, 0, &prefilter, &init, &options).unwrap();
            let b0 = result.parameter[0];
            println!("q = {q}: b_0 = {b0:.4} (1 / J = {}), {} iterations", 1.0 / j, result.iterations);
            assert!(result.converged && result.iterations == iterations && result.parameter.len() == 1);
            estimates.push(b0);
            assert!((b0 * j - 1.0).abs() < 2e-3, "q = {q}: b_0 = {b0}");
            let mut expected = vec![1.0];
            expected.resize(q + 1, 0.0);
            assert_eq!(result.model.tf.numerator.0, vec![b0]);
            assert_eq!(result.model.tf.denominator.0, expected);
        }
        assert!((estimates[0] / estimates[1] - 1.0).abs() < 1e-12, "{estimates:?}");
    }
}

/// Rigid-body mode alone of a two-inertia system (torque -> position) with a resonance at 1.5 kHz,
/// well above the excited band (10 ..= 200 Hz): `1 / (s (J s + D)) · ωr^2 / (s^2 + 2 ζ ωr s + ωr^2)`
/// with a small `D` (pole at `D / J` = 0.1 rad/s). `(0, 0, 2)` gives `b_0 ≈ 1 / J`, biased by the
/// quasi-static tail of the resonance, `(f / fr)^2` (< 2 % in the band).
#[test]
fn test_srivc_pure_integrator_rigid_body() {
    let (j, d) = (0.01, 0.001);
    let wr = 2.0 * PI * 1500.0;
    let rigid = TransferFunction::continuous(&[1.0], &[j, d, 0.0]);
    let plant = TransferFunctionWithDelay::from(&rigid * &TransferFunction::continuous(&[wr * wr], &[1.0, 2.0 * 0.05 * wr, wr * wr]));
    let (u, y) = experiment(&plant, 3 * PERIOD, 0x1234_5678_9abc, 0.05, 0.0);
    let options = SrivcOptions { evaluated_from: PERIOD, ..SrivcOptions::default() };
    let prefilter = Prefilter::new(2, CUTOFF).unwrap();
    let init = Initialization::StateVariableFilter(2.0 * PI * 100.0);
    let result = identify_with_prefilter(&u, &y, TS, 0, 0, &prefilter, &init, &options).unwrap();
    let j_hat = 1.0 / result.parameter[0];
    println!("J = {j_hat:.5} (true {j}), {:+.2} %", 100.0 * (j_hat / j - 1.0));
    assert!(result.converged && (j_hat / j - 1.0).abs() < 0.02, "J = {j_hat}");
}

/// `n = 0` without integrators (a static gain) is `InvalidOrder`, by `identify` and by a
/// prefilter with `q = 0`.
#[test]
fn test_srivc_pure_integrator_invalid_order() {
    let u: Vec<f64> = (0..1000).map(|k| (k as f64 * 0.1).sin()).collect();
    let init = Initialization::StateVariableFilter(100.0);
    let options = SrivcOptions::default();
    let invalid = Err(SrivcError::InvalidOrder { denominator: 0, numerator: 0 });
    assert_eq!(identify(&u, &u, TS, 0, 0, &init, &options).map(|_| ()), invalid);
    let prefilter = Prefilter::new(0, CUTOFF).unwrap();
    assert_eq!(identify_with_prefilter(&u, &u, TS, 0, 0, &prefilter, &init, &options).map(|_| ()), invalid);
    assert_eq!(
        identify_with_prefilter(&u, &u, TS, 0, 1, &Prefilter::new(2, CUTOFF).unwrap(), &init, &options).map(|_| ()),
        Err(SrivcError::InvalidOrder { denominator: 0, numerator: 1 })
    );
}

/// `search` over `Structure::grid(0..=2, 0..=2, 0..=2, 0..=0)` on `1 / (J s^2)`: every candidate
/// is run, `(0, 0, 0)` is not identified (`InvalidOrder`), and `(0, 0, 2)` (one parameter) is
/// selected by BIC over the structures that fit as well with more parameters.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_pure_integrator_search() {
    let j = 0.01;
    let plant = TransferFunctionWithDelay::from(TransferFunction::continuous(&[1.0], &[j, 0.0, 0.0]));
    let (u, y) = experiment(&plant, 3 * PERIOD, 0x1234_5678_9abc, 0.05, 0.0);
    let (u_val, y_val) = experiment(&plant, 4 * PERIOD, 0x0fed_cba9_8765, 0.05, 0.0);
    let structures = Structure::grid(0..=2, 0..=2, 0..=2, 0..=0);
    let options = SearchOptions {
        evaluated_from: PERIOD,
        prefilter: Some(CUTOFF),
        srivc: SrivcOptions { evaluated_from: PERIOD, ..SrivcOptions::default() },
        ..SearchOptions::new(
            Initialization::StateVariableFilter(2.0 * PI * 100.0),
            vec![
                Check::CrossCorrelation { max_lag: 100 },
                Check::Lines { fundamental_frequency: F0, lines: BAND.collect() },
                Check::Coherence(CoherenceCheck::new(1.0 / F0, 1e-2)),
            ],
            0.99,
        )
    };
    let search = srivc::search((&u, &y), (&u_val, &y_val), TS, &structures, &options).unwrap();
    println!("{search}");

    assert_eq!(search.candidates.len(), structures.len());
    let candidate = |n, m, q| search.candidates.iter().find(|c| c.structure == Structure::new(n, m, q, 0)).unwrap();
    assert!(matches!(candidate(0, 0, 0).outcome, Outcome::NotIdentified(SrivcError::InvalidOrder { denominator: 0, numerator: 0 })));
    assert!(candidate(0, 0, 1).report().is_some());
    let selected = search.selected().unwrap();
    assert_eq!(selected.structure, Structure::new(0, 0, 2, 0));
    let b0 = selected.result().unwrap().parameter[0];
    assert!((b0 * j - 1.0).abs() < 2e-3, "b_0 = {b0}");
}
