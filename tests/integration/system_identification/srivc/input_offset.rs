//! An input offset not in the recorded input, and the high-pass filter that removes it.

use std::f64::consts::PI;

use dsmc::system_identification::iv::srivc::{identify, Initialization, SrivcOptions};
use dsmc::system_identification::preprocessing::high_pass;

use super::*;

/// A constant input offset that is not in the recorded input (a torque bias in an open-loop test)
/// is integrated by the plant into a ramp of the output. SRIVC minimizes the output error, so the
/// ramp dominates the fit: an offset of 1 % of the input RMS gives a relative error of ~9 %
/// (0.37 % without it; 0.1 % is still within the noise; the size depends on the multisine
/// phases).
/// The same high-pass filter (3 Hz, twice, below the excited band) on the input and the output
/// (`preprocessing::high_pass`) restores the estimate.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_input_offset() {
    let plant = two_inertia_plant();
    let (m, n) = orders(&plant);
    let options = SrivcOptions { input_delay: 8, ..SrivcOptions::default() };
    let init = Initialization::StateVariableFilter(2.0 * PI * 100.0);
    let antiresonance = 36..=44;

    for offset in [0.0, 0.001, 0.01] {
        let (u, y) = experiment(&plant, 4 * PERIOD, 0x1234_5678_9abc, 0.05, offset);

        let raw = identify(&u, &y, TS, n, m, &init, &options).unwrap();
        let (raw_error, raw_phase) = band_error(&raw.model, &plant, &antiresonance);

        let (uf, yf) = high_pass(&u, &y, 3.0, TS, 2);
        let filtered = identify(&uf, &yf, TS, n, m, &init, &options).unwrap();
        let (error, phase) = band_error(&filtered.model, &plant, &antiresonance);

        println!("offset {offset}: raw {raw_error:.4} / {raw_phase:.2} deg, high-passed {error:.4} / {phase:.2} deg");
        assert!(filtered.converged && error < 0.02 && phase < 1.0, "offset {offset}: high-passed, error {error:e}, phase {phase} deg");
        if offset == 0.0 {
            // Without a drift the filter costs nothing
            assert!(raw.converged && raw_error < 0.02);
        }
        if offset == 0.01 {
            assert!(raw_error > 10.0 * error, "offset {offset}: raw {raw_error:e}, high-passed {error:e}");
        }
    }
}
