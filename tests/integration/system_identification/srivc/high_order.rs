//! A high order (8), regression test of the scaling of the SRIVC prefilter.

use std::f64::consts::PI;

use dsmc::system_identification::iv::srivc::{identify, Initialization, SrivcOptions};
use dsmc::system_identification::preprocessing::high_pass;

use super::*;

/// Order 8 (three inertias, position output, low-pass): SRIVC converges in a few iterations (4
/// without noise, 8 with), to an error of ~1e-8 without noise.
///
/// Regression test of the scaling of the SRIVC prefilter: the coefficients `a_i` of `A(s)` grow
/// like `ω^i`, so here the companion matrix spans ~22 decades. Before the states of the prefilter
/// were scaled by the root radius of `A(s)`, its matrix exponential was inaccurate and the
/// iterations ran into the limit (20) with an error of ~3e-2 without noise (checked by disabling
/// the scaling), which fails the tolerances below. The lower-order tests do not span enough
/// decades to catch that.
///
/// The position of a free body drifts in an open-loop test, so the signals are high-passed as in
/// `test_srivc_input_offset`.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_high_order() {
    let plant = three_inertia_plant();
    let (m, n) = orders(&plant);
    let options = SrivcOptions { input_delay: 8, ..SrivcOptions::default() };
    let init = Initialization::StateVariableFilter(2.0 * PI * 100.0);
    let nothing = 0..=0;

    for noise_ratio in [0.0, 0.05] {
        // Noise relative to the high-passed output (the raw position is dominated by its ramp)
        let (u, y0) = experiment(&plant, 4 * PERIOD, 0x1234_5678_9abc, 0.0, 0.0);
        let (u, y0) = high_pass(&u, &y0, 3.0, TS, 2);
        let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
        let scale = noise_ratio * rms(&y0);
        let y: Vec<f64> = y0.iter().map(|v| v + scale * rng.gaussian()).collect();

        let result = identify(&u, &y, TS, n, m, &init, &options).unwrap();
        let (error, phase) = band_error(&result.model, &plant, &nothing);
        println!(
            "noise {noise_ratio}: {} iterations{}, relative error {error:.3e}, phase error {phase:.3} deg",
            result.iterations,
            if result.converged { "" } else { " (not converged)" },
        );
        let tolerance = if noise_ratio == 0.0 { 1e-4 } else { 5e-2 };
        assert!(result.converged && result.iterations <= 10 && error < tolerance, "noise {noise_ratio}: error {error:e}");
    }
}
