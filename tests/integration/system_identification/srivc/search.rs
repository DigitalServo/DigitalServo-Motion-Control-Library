//! Search of the structure `(n, m, q, nk)` by `srivc::search`.

use std::f64::consts::PI;

use dsmc::system_identification::iv::srivc::{self, Initialization, Outcome, SearchOptions, Structure};
use dsmc::system_identification::validation::{Check, CheckKind};

use super::*;

/// Joint search of `(n, m, nk)` by `srivc::search`: each candidate is identified on one
/// experiment and validated on another one (different multisine phases and noise) at 99 % by
/// the information criteria, whiteness, cross-correlation, the test at the excited lines and
/// coherence; the converged candidates passing every test are compared by BIC.
///
/// - Missing poles are made up for by a longer delay, so the best delay depends on the order
///   (16 samples for `n = 3`, 13 for `n = 4`, the true 8 for `n = 5`): the delay cannot be fixed
///   first and the orders searched afterwards.
/// - The validation errors `V` of all reasonable candidates agree to 3 digits, while the
///   cross-correlation test rejects every lower order and every wrong delay: the residual still
///   depends on the input. Some of them are unstable instead (a pole in the right half-plane
///   making up for the missing dynamics), and are left out before the tests.
/// - The line test also rejects the lower orders (mean F several times its expectation), but is
///   less sharp for a delay off by one sample.
/// - With too many parameters the delay is no longer identifiable (an extra zero absorbs a shift),
///   and the iterations often do not converge; the converged ones pass every test, and lose in
///   BIC by about the penalty `ln N` of the extra parameter.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_structure_search() {
    let plant = two_inertia_plant();
    let noise_ratio = 0.05;
    let (u, y) = experiment(&plant, 4 * PERIOD, 0x1234_5678_9abc, noise_ratio, 0.0);
    // Validation: 6 periods, the first one (transient from rest) not evaluated
    let (u_val, y_val) = experiment(&plant, 6 * PERIOD, 0x0fed_cba9_8765, noise_ratio, 0.0);
    let antiresonance = 36..=44;

    let structures: Vec<Structure> = [(3, 2, vec![14, 16, 18]), (4, 2, vec![12, 13, 14]), (5, 2, vec![6, 7, 8, 9, 10]), (5, 3, vec![6, 8, 10]), (6, 3, vec![8])]
        .into_iter()
        .flat_map(|(n, m, delays)| delays.into_iter().map(move |nk| Structure::new(n, m, 0, nk)))
        .collect();
    let options = SearchOptions {
        evaluated_from: PERIOD,
        ..SearchOptions::new(
            Initialization::StateVariableFilter(2.0 * PI * 100.0),
            vec![
                Check::Whiteness { max_lag: 20 },
                Check::CrossCorrelation { max_lag: 100 },
                Check::Lines { fundamental_frequency: F0, lines: BAND.collect() },
                Check::Coherence { segment_s: 1.0 / F0, excited: 1e-2 },
            ],
            0.99,
        )
    };
    let search = srivc::search((&u, &y), (&u_val, &y_val), TS, &structures, &options).unwrap();
    println!("{search}");
    for c in &search.candidates {
        if let Some(result) = c.result() {
            let (relative, phase) = band_error(&result.model, &plant, &antiresonance);
            println!("{}: max relative error {relative:.4}, max phase error {phase:.2} deg", c.structure);
        }
    }

    let selected = search.selected().unwrap();
    println!("{}", selected.report().unwrap());
    assert_eq!(selected.structure, Structure::new(5, 2, 0, 8));
    let (relative, phase) = band_error(&selected.result().unwrap().model, &plant, &antiresonance);
    assert!(relative < 0.02 && phase < 1.0, "selected model: relative error {relative:e}, phase error {phase} deg");

    let validated = || search.candidates.iter().filter_map(|c| c.report().map(|r| (c.structure, r)));
    // The lower orders and the wrong delays fail the cross-correlation test, the lower orders
    // also the line test, by a mean F well above its expectation
    for (s, report) in validated().filter(|(s, _)| s.denominator_order < 5 || (s.numerator_order == 2 && s.input_delay != 8)) {
        assert_eq!(report.passed_by(CheckKind::CrossCorrelation), Some(false), "{s} not rejected by the cross-correlation");
    }
    for (s, report) in validated().filter(|(s, _)| s.denominator_order < 5) {
        let lines = report.lines().unwrap();
        let ratio = lines.mean_statistic() / lines.expected_statistic();
        assert!(report.passed_by(CheckKind::Lines) == Some(false) && ratio > 2.0, "{s}: line test, mean F {ratio}");
    }

    // An extra zero passes the tests when converged, but loses in BIC
    let extra: Vec<_> = search.candidates.iter().filter(|c| c.structure.numerator_order == 3 && c.selectable()).collect();
    assert!(!extra.is_empty() && extra.iter().all(|c| c.bic() > selected.bic()));
    // Every candidate is identified; the unstable ones (a pole in the right half-plane making
    // up for a missing pole or a wrong delay) are lower orders or wrong delays
    let wrong = |s: &Structure| s.denominator_order < 5 || s.input_delay != 8;
    for c in &search.candidates {
        assert!(matches!(c.outcome, Outcome::Validated { .. }) || matches!(c.outcome, Outcome::Unstable(_)) && wrong(&c.structure), "{}", c.structure);
    }
}

/// Joint search of `(n, m, q, nk)` with the prefilters at `ω_c` = 3 Hz (order `q + 1` for each
/// candidate, 3 on the validation data), on
/// data with an input offset of 1 % (not recorded): the right number of integrators `q = 1`
/// is selected.
///
/// - A wrong `q` with the orders of the right one fails every test or is unstable: with `q = 0`
///   the pole at the origin is missing from `A`, with `q = 2` the zero at the origin `s^2 G`
///   needs is missing from `B`.
/// - Too many integrators with the orders adjusted (`(4, 3, 2)`: a zero near the origin) fits as
///   well and passes every test, but loses in BIC by about the penalty of the extra parameter,
///   `ln N` (~11).
/// - Too few with the orders adjusted (`(5, 2, 0)`: a pole near the origin) would lose the same
///   way without the offset, but here its prefilter (order `q + 1 = 1`) leaves the ramp of the
///   output: the iterations do not converge.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_integrator_search() {
    let plant = two_inertia_plant();
    let (noise_ratio, offset) = (0.05, 0.01);
    let (u, y) = experiment(&plant, 4 * PERIOD, 0x1234_5678_9abc, noise_ratio, offset);
    let (u_val, y_val) = experiment(&plant, 6 * PERIOD, 0x0fed_cba9_8765, noise_ratio, offset);
    let antiresonance = 36..=44;

    let mut structures = Vec::new();
    for (n, m, q) in [(4, 2, 0), (5, 2, 0), (3, 2, 1), (4, 2, 1), (4, 3, 1), (5, 2, 1), (4, 2, 2), (4, 3, 2), (3, 3, 2)] {
        for nk in [7, 8, 9] {
            structures.push(Structure::new(n, m, q, nk));
        }
    }
    let options = SearchOptions {
        evaluated_from: PERIOD,
        prefilter: Some(2.0 * PI * 3.0),
        ..SearchOptions::new(
            Initialization::StateVariableFilter(2.0 * PI * 100.0),
            vec![
                Check::Whiteness { max_lag: 20 },
                Check::CrossCorrelation { max_lag: 100 },
                Check::Lines { fundamental_frequency: F0, lines: BAND.collect() },
                Check::Coherence { segment_s: 1.0 / F0, excited: 1e-2 },
            ],
            0.99,
        )
    };
    let search = srivc::search((&u, &y), (&u_val, &y_val), TS, &structures, &options).unwrap();
    println!("{search}");
    for c in &search.candidates {
        if let Some(result) = c.result() {
            let (relative, phase) = band_error(&result.model, &plant, &antiresonance);
            println!("{}: max relative error {relative:.4}, max phase error {phase:.2} deg", c.structure);
        }
    }

    let selected = search.selected().unwrap();
    assert_eq!(selected.structure, Structure::new(4, 2, 1, 8));
    let (relative, phase) = band_error(&selected.result().unwrap().model, &plant, &antiresonance);
    assert!(relative < 0.02 && phase < 1.0, "selected model: relative error {relative:e}, phase error {phase} deg");

    let candidate = |n, m, q, nk| search.candidates.iter().find(|c| c.structure == Structure::new(n, m, q, nk)).unwrap();
    for nk in [7, 8, 9] {
        for (n, m, q) in [(4, 2, 0), (4, 2, 2)] {
            let c = candidate(n, m, q, nk);
            assert!(c.report().is_none_or(|r| !r.passed()) && !c.selectable(), "{} passes", c.structure);
        }
    }
    let c = candidate(4, 3, 2, 8);
    assert!(c.selectable() && c.bic().unwrap() > selected.bic().unwrap() + 5.0, "{}: BIC {:?}", c.structure, c.bic());
    assert!(!candidate(5, 2, 0, 8).result().unwrap().converged);
}
