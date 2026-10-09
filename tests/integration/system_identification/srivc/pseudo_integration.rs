//! Integrators of the plant through the pseudo-integrating prefilter (`identify_with_prefilter`).

use std::f64::consts::PI;

use dsmc::system_identification::iv::srivc::{identify_with_prefilter, Initialization, Prefilter, SrivcError, SrivcOptions};

use super::*;

/// `srivc::identify_with_prefilter`: `R(s) = s G(s)` (order 4, finite gain) from the
/// pseudo-integrated input `s / (s + ω_c)^2 u` and the high-passed output `s^2 / (s + ω_c)^2 y`,
/// with `ω_c` = 3 Hz (below the excited band, which starts at 10 Hz), instead of a fifth-order
/// `G(s)` whose pole near the origin is not in the data (it lands anywhere around 1e-3 rad/s).
///
/// - The model returned, `R(s) / s` (the pole exactly at the origin), matches the plant over the
///   band, and the rigid-body gain `R(0)` (= `dc_gain`, the low-frequency asymptote `R(0) / s`)
///   is found from the parameters of `R`.
/// - An input offset of 1 % (the ramp of the output) does not need a separate high-pass: the
///   second-order prefilter removes it.
/// - Without noise the error is ~5e-5, not ~1e-10 as by `identify`: the output is filtered as
///   linear between samples, and the error of that interpolation no longer cancels at
///   convergence once the high-pass `s^k / (s + ω_c)^k` is in the filter; it grows like
///   `k ω_c ts^2` (~1.5e-5 per Hz of the cutoff and per order here), far below the noise.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_pseudo_integration() {
    let plant = two_inertia_plant();
    let options = SrivcOptions { input_delay: 8, ..SrivcOptions::default() };
    let init = Initialization::StateVariableFilter(2.0 * PI * 100.0);
    let prefilter = Prefilter::new(1, 2.0 * PI * 3.0).unwrap();
    let antiresonance = 36..=44;

    for (noise_ratio, offset, tolerance) in [(0.0, 0.0, 1e-4), (0.05, 0.0, 0.01), (0.05, 0.01, 0.01)] {
        let (u, y) = experiment(&plant, 4 * PERIOD, 0x1234_5678_9abc, noise_ratio, offset);
        let result = identify_with_prefilter(&u, &y, TS, 4, 2, &prefilter, &init, &options).unwrap();
        // G(s) = R(s) / s, with R(0) = b_2 / a_4 (θ = [a_1, ..., a_4, b_0, b_1, b_2])
        let (relative, phase) = band_error(&result.model, &plant, &antiresonance);
        let rigid_body_gain = result.parameter[6] / result.parameter[3];
        println!("noise {noise_ratio}, offset {offset}: relative error {relative:.2e}, phase error {phase:.3} deg, R(0) {rigid_body_gain:.3}, {} iterations", result.iterations);
        println!("tf: {}", result.model);
        assert!(result.converged && orders(&result.model) == (2, 5) && result.model.tf.denominator[5] == 0.0);
        assert!(relative < tolerance && phase < 100.0 * tolerance, "noise {noise_ratio}, offset {offset}: relative error {relative:e}, phase error {phase} deg");
        assert!((rigid_body_gain / 100.0 - 1.0).abs() < tolerance, "R(0) = {rigid_body_gain}");
    }

    assert_eq!(prefilter.order(), 2);
    for cutoff in [0.0, -10.0, f64::NAN, f64::INFINITY] {
        assert_eq!(Prefilter::new(1, cutoff), Err(SrivcError::InvalidPrefilter));
    }
}

/// `srivc::identify_with_prefilter` with two integrators (three inertias, torque -> position):
/// `R(s) = s^2 G(s)` of order 6 through the prefilter of order 3 at 3 Hz.
///
/// An input offset not in the recorded input (a ramp `~ t^2` of the output) is removed by the
/// prefilter but for its transient from `k = 0`, `c R(s) / (s + ω_c)^3`, decaying with `ω_c`
/// and the lightly damped modes (`1 / (ζ ω)` ~ 0.1 s): without noise it biases the estimate in
/// proportion to the offset (an offset of 10 % does not even converge), and leaving out the
/// first period of the sums (`evaluated_from`) removes the bias. With 5 % noise and the first
/// period left out, the error does not depend on the offset.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_pseudo_integration_two_integrators() {
    let plant = three_inertia_plant();
    let prefilter = Prefilter::new(2, 2.0 * PI * 3.0).unwrap();
    let init = Initialization::StateVariableFilter(2.0 * PI * 100.0);
    let (n, m) = (6, 4);
    let nothing = 0..=0;
    let run = |noise_ratio: f64, offset: f64, evaluated_from: usize| {
        let (u, y) = experiment(&plant, 4 * PERIOD, 0x1234_5678_9abc, noise_ratio, offset);
        let options = SrivcOptions { input_delay: 8, evaluated_from, ..SrivcOptions::default() };
        let result = identify_with_prefilter(&u, &y, TS, n, m, &prefilter, &init, &options).unwrap();
        let (relative, phase) = band_error(&result.model, &plant, &nothing);
        // R(0) = b_4 / a_6, θ = [a_1, ..., a_6, b_0, ..., b_4]
        let rigid_body_gain = result.parameter[n + m] / result.parameter[n - 1];
        println!(
            "noise {noise_ratio}, offset {offset}, evaluated from {evaluated_from}: relative error {relative:.2e}, phase error {phase:.3} deg, R(0) {rigid_body_gain:.1}, converged {}",
            result.converged
        );
        assert_eq!(orders(&result.model), (4, 8));
        (relative, result.converged, rigid_body_gain)
    };

    for offset in [0.0, 0.01, 0.1] {
        let (all, all_converged, _) = run(0.0, offset, 0);
        let (relative, converged, rigid_body_gain) = run(0.0, offset, PERIOD);
        assert!(converged && relative < 1e-4 && (rigid_body_gain / 1e4 - 1.0).abs() < 1e-4, "offset {offset}: relative error {relative:e}, R(0) {rigid_body_gain}");
        if offset == 0.1 {
            assert!(!all_converged || all > 100.0 * relative, "offset {offset}: from 0 {all:e}, from one period {relative:e}");
        }
    }

    let (without, ..) = run(0.05, 0.0, PERIOD);
    for offset in [0.01, 0.1] {
        let (relative, converged, _) = run(0.05, offset, PERIOD);
        assert!(converged && (relative / without - 1.0).abs() < 0.05, "offset {offset}: relative error {relative:e} (without offset {without:e})");
    }
}
