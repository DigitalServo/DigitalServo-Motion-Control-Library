//! SRIVC on plants typical of an open-loop motion-control identification test: multi-inertia
//! mechanics (rigid-body integrator, lightly damped resonances / antiresonances) seen through an
//! analog low-pass filter and a dead time, excited by a band-limited multisine.
//!
//! Three things are checked, each of them needed to use `srivc::identify` on such data:
//!
//! 1. `test_srivc_structure_search`: the denominator order `n`, the numerator order `m` and the
//!    input delay `nk` have to be searched *jointly*, and validated by the residual tests of
//!    `Validation` (not by the raw validation error, which differs only in the 4th digit between
//!    candidates), then compared by BIC.
//! 2. `test_srivc_input_offset`: a small constant input offset (drift of the integrator) ruins the
//!    estimate; the same high-pass filter on the input and the output removes the problem.
//! 3. `test_srivc_time_normalization`: at high orders the coefficients of `A(s)` span many
//!    decades; `srivc::identify` scales the prefilter internally, so the plain call converges and
//!    agrees with an explicit time normalization (`identify_normalized`).
//!
//! The tests take ~10 s with optimizations and several minutes without, so they are ignored in
//! debug builds: run `cargo test --release --test srivc -- --nocapture` (the
//! tables of the candidates are printed), or force them with `-- --ignored`.

use std::f64::consts::PI;

use dsmc::TransferFunction;
use dsmc::discretize::exact_discretize::DiscretizedSystem;
use dsmc::system_identification::iv::srivc::{Initialization, SrivcOptions, identify};
use dsmc::system_identification::validation::Validation;
use num_complex::Complex;

const TS: f64 = 1e-4;
/// Excited band of the multisine \[Hz\] (1 Hz spacing, so the period is 1 s).
const BAND: std::ops::RangeInclusive<usize> = 10..=200;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// xorshift64 pseudo-random numbers.
struct Xorshift(u64);

impl Xorshift {
    /// Uniform in (0, 1).
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    fn gaussian(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (2.0 * PI * self.uniform()).cos()
    }
}

/// Product of two polynomials (coefficients in descending order).
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            c[i + j] += x * y;
        }
    }
    c
}

/// `s^2 + 2 ζ ω s + ω^2` with `ω = 2π f`.
fn second_order(f: f64, zeta: f64) -> Vec<f64> {
    let w = 2.0 * PI * f;
    vec![1.0, 2.0 * zeta * w, w * w]
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

/// Continuous-time model `e^(-delay ts s) B(s) / A(s)` (coefficients in descending order).
#[derive(Clone, Debug)]
struct Model {
    numerator: Vec<f64>,
    denominator: Vec<f64>,
    /// Input delay \[samples\].
    delay: usize,
}

impl Model {
    /// `G(jω)` including the dead time.
    fn response(&self, omega: f64) -> Complex<f64> {
        let s = Complex::new(0.0, omega);
        let eval = |c: &[f64]| c.iter().fold(Complex::new(0.0, 0.0), |acc, &x| acc * s + x);
        eval(&self.numerator) / eval(&self.denominator) * Complex::new(0.0, -omega * self.delay as f64 * TS).exp()
    }

    /// Sampled response to the input `u` applied through a zero-order hold, from rest.
    /// `None` if the model cannot be realized or the response diverges (unstable model).
    fn simulate(&self, u: &[f64]) -> Option<Vec<f64>> {
        let tf = TransferFunction::continuous(&self.numerator, &self.denominator);
        let mut system = DiscretizedSystem::from_tf(&tf, TS).ok()?;
        let y: Vec<f64> = (0..u.len())
            .map(|k| system.update(&[if k >= self.delay { u[k - self.delay] } else { 0.0 }]).unwrap()[0])
            .collect();
        y.iter().all(|v| v.is_finite()).then_some(y)
    }
}

/// Multisine over `BAND` with random phases and unit RMS, starting at `k = 0` (zero before).
fn multisine(n_samples: usize, seed: u64) -> Vec<f64> {
    let mut rng = Xorshift(seed);
    let phases: Vec<f64> = BAND.map(|_| 2.0 * PI * rng.uniform()).collect();
    let scale = (2.0 / phases.len() as f64).sqrt();
    (0..n_samples)
        .map(|k| {
            let t = k as f64 * TS;
            scale * BAND.zip(&phases).map(|(f, p)| (2.0 * PI * f as f64 * t + p).sin()).sum::<f64>()
        })
        .collect()
}

/// Open-loop experiment from rest: multisine input `u`, and the output of `plant` to
/// `u + input_offset` plus white noise of standard deviation `noise_ratio` times the RMS of the
/// noise-free output (about its mean). The offset is not part of the recorded input.
fn experiment(plant: &Model, n_samples: usize, seed: u64, noise_ratio: f64, input_offset: f64) -> (Vec<f64>, Vec<f64>) {
    let u = multisine(n_samples, seed);
    let applied: Vec<f64> = u.iter().map(|x| x + input_offset).collect();
    let y0 = plant.simulate(&applied).unwrap();
    let mean = y0.iter().sum::<f64>() / n_samples as f64;
    let scale = noise_ratio * rms(&y0.iter().map(|v| v - mean).collect::<Vec<_>>());
    let mut rng = Xorshift(seed ^ 0x0abc_def1_2345);
    let y = y0.iter().map(|v| v + scale * rng.gaussian()).collect();
    (u, y)
}

/// First-order high-pass filter with cutoff `fc` \[Hz\], at rest before `k = 0`.
///
/// To remove a drift it has to be applied to the input *and* the output: the plant is linear, so
/// the filtered signals are still related by the same `G(s)`, and a filtered sampled input is
/// still constant between samples, so the zero-order-hold assumption of SRIVC stays exact.
///
/// The first sample matters: `x[-1] = 0` here, i.e. the step of the signal at `k = 0` is filtered
/// too. Starting from `x[-1] = x[0]` instead breaks the relation between the two filtered signals
/// (an error of 5 .. 20 % on the plant below).
fn high_pass(x: &[f64], fc: f64) -> Vec<f64> {
    let a = 1.0 / (1.0 + 2.0 * PI * fc * TS);
    let mut out = vec![0.0; x.len()];
    let (mut previous_in, mut previous_out) = (0.0, 0.0);
    for (k, &v) in x.iter().enumerate() {
        out[k] = a * (previous_out + v - previous_in);
        previous_in = v;
        previous_out = out[k];
    }
    out
}

/// Result of `identify_normalized`.
struct Fit {
    model: Model,
    iterations: usize,
    converged: bool,
}

/// SRIVC in normalized time `t' = w0 t` (`s' = s / w0`), `w0` \[rad/s\] being a frequency inside
/// the band of interest; `w0 = 1` is the plain call. The coefficients of `A(s)` grow like
/// `ω^i` and span many decades at high orders; `identify` scales its prefilter by the root radius
/// of `A(s)` itself, so the normalization is not needed for convergence (see
/// `test_srivc_time_normalization`) and only changes the rounding.
///
/// The fitted `G'(s') = Σ b'_j s'^(m-j) / (s'^n + Σ a'_i s'^(n-i))` is `G(s) = G'(s / w0)`:
/// `a_i = a'_i w0^i`, `b_j = b'_j w0^(n-m+j)`.
///
/// `svf` is the bandwidth \[rad/s\] of the state-variable filter starting the iterations.
fn identify_normalized(u: &[f64], y: &[f64], m: usize, n: usize, delay: usize, svf: f64, w0: f64) -> Option<Fit> {
    let options = SrivcOptions { input_delay: delay, max_iterations: 20, ..SrivcOptions::default() };
    let result = identify(u, y, TS * w0, m, n, &Initialization::StateVariableFilter(svf / w0), &options).ok()?;
    let theta = result.parameter.as_slice();
    if theta.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let mut denominator = vec![1.0];
    denominator.extend((1..=n).map(|i| theta[i - 1] * w0.powi(i as i32)));
    let numerator = (0..=m).map(|j| theta[n + j] * w0.powi((n - m + j) as i32)).collect();
    Some(Fit { model: Model { numerator, denominator, delay }, iterations: result.iterations, converged: result.converged })
}

/// Model error against the true plant over `BAND`, leaving out `exclude` \[Hz\] (the notch of an
/// antiresonance, where the relative error is dominated by the tiny gain):
/// (max relative error `|G - G0| / |G0|`, max phase error \[deg\]).
fn band_error(model: &Model, plant: &Model, exclude: &std::ops::RangeInclusive<usize>) -> (f64, f64) {
    BAND.filter(|f| !exclude.contains(f)).fold((0.0, 0.0), |(relative, phase): (f64, f64), f| {
        let omega = 2.0 * PI * f as f64;
        let (g, g0) = (model.response(omega), plant.response(omega));
        (relative.max((g - g0).norm() / g0.norm()), phase.max((g / g0).arg().abs().to_degrees()))
    })
}

// ---------------------------------------------------------------------------------------------
// Plants
// ---------------------------------------------------------------------------------------------

/// Two-inertia system, torque -> velocity (antiresonance 40 Hz, resonance 65 Hz, ζ = 0.02),
/// second-order Butterworth low-pass at 300 Hz in the measurement, dead time of 8 samples (0.8 ms):
/// `n = 5`, `m = 2`, `nk = 8`.
fn two_inertia_plant() -> Model {
    let (antiresonance, resonance, low_pass) = (second_order(40.0, 0.02), second_order(65.0, 0.02), second_order(300.0, 0.5f64.sqrt()));
    // Rigid-body gain 100 / s at low frequency
    let gain = 100.0 * resonance[2] / antiresonance[2] * low_pass[2];
    Model {
        numerator: antiresonance.iter().map(|c| c * gain).collect(),
        denominator: convolve(&convolve(&[1.0, 0.0], &resonance), &low_pass),
        delay: 8,
    }
}

/// Three-inertia system, torque -> position (antiresonances 40 / 90 Hz, resonances 65 / 120 Hz),
/// the same low-pass and dead time: `n = 8`, `m = 4`, `nk = 8`.
fn three_inertia_plant() -> Model {
    let zeros = convolve(&second_order(40.0, 0.02), &second_order(90.0, 0.02));
    let poles = convolve(&second_order(65.0, 0.02), &second_order(120.0, 0.02));
    let low_pass = second_order(300.0, 0.5f64.sqrt());
    let gain = 1e4 * poles[4] / zeros[4] * low_pass[2];
    Model {
        numerator: zeros.iter().map(|c| c * gain).collect(),
        denominator: convolve(&convolve(&[1.0, 0.0, 0.0], &poles), &low_pass),
        delay: 8,
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

/// Joint search of `(n, m, nk)`: each candidate is identified on one experiment and validated on
/// another one (different multisine phases and noise) with `Validation`:
///
/// - BIC `N ln V + p ln N` (`V`: mean squared simulation error, `p = n + m + 1`),
/// - cross-correlation test of the residual and the input (`τ = -100 ..= 100`, 99 % per lag),
/// - test at the excited lines of the multisine (model error against the noise level estimated
///   per line from the period-to-period variation, 99 % per line).
///
/// The candidates passing both tests (at most 5 % of the lags / lines outside, for a nominal 1 %)
/// are compared by BIC, and the lowest is selected.
///
/// - Missing poles are made up for by a longer delay, so the best delay depends on the order
///   (16 samples for `n = 3`, 13 for `n = 4`, the true 8 for `n = 5`): the delay cannot be fixed
///   first and the orders searched afterwards.
/// - The validation errors `V` of all reasonable candidates agree to 3 digits, while the
///   cross-correlation test rejects every lower order and every wrong delay (40 .. 80 % of the
///   lags outside): the residual still depends on the input.
/// - The line test is less sharp with 5 periods at 99 % (a delay off by one sample passes with
///   1 .. 3 % of the lines outside, mean F 1.2 .. 1.3 times its expectation), but rejects the lower
///   orders by their mean F (2 .. 70 times its expectation).
/// - With too many parameters the delay is no longer identifiable (an extra zero absorbs a shift),
///   and the iterations often do not converge; the converged ones pass both tests, and lose in BIC
///   by about the penalty `ln N` of the extra parameter.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_structure_search() {
    let plant = two_inertia_plant();
    let period = (1.0 / TS).round() as usize;
    let noise_ratio = 0.05;
    let (u, y) = experiment(&plant, 4 * period, 0x1234_5678_9abc, noise_ratio, 0.0);
    // Validation: 6 periods, the first one (transient from rest) not used by the line test
    let (u_val, y_val) = experiment(&plant, 6 * period, 0x0fed_cba9_8765, noise_ratio, 0.0);
    let lines: Vec<usize> = BAND.collect();

    let w0 = 2.0 * PI * 60.0;
    let svf = 2.0 * PI * 100.0;
    let antiresonance = 36..=44;

    // (n, m, delays)
    let candidates: [(usize, usize, &[usize]); 5] =
        [(3, 2, &[14, 16, 18]), (4, 2, &[12, 13, 14]), (5, 2, &[6, 7, 8, 9, 10]), (5, 3, &[6, 8, 10]), (6, 3, &[8])];

    struct Row {
        n: usize,
        m: usize,
        fit: Fit,
        bic: f64,
        /// Cross-correlation test: fraction of the lags outside, max |r| / bound
        correlation: (f64, f64),
        /// Line test: fraction of the lines outside, mean F / expected F
        lines: (f64, f64),
    }
    impl Row {
        fn passes(&self) -> bool {
            self.fit.converged && self.correlation.0 <= 0.05 && self.lines.0 <= 0.05
        }
    }
    let mut rows = Vec::new();
    println!(" n  m nk |      BIC | corr. out  max |  lines out  mean F | max rel. err | max phase err | iterations");
    for (n, m, delays) in candidates {
        for &delay in delays {
            let Some(fit) = identify_normalized(&u, &y, m, n, delay, svf, w0) else { continue };
            let model = TransferFunction::continuous(&fit.model.numerator, &fit.model.denominator);
            let validation = Validation::continuous(&model, delay, TS, &u_val, &y_val).unwrap();
            if !validation.mse().is_finite() {
                println!("{n:2} {m:2} {delay:2} | unstable model ({} iterations)", fit.iterations);
                continue;
            }
            let bic = validation.bic(n + m + 1);
            let correlation = validation.cross_correlation(100, 2.58);
            let line_test = validation.clone().evaluated_from(period).line_test(period, &lines, 0.99).unwrap();
            let row = Row {
                n,
                m,
                bic,
                correlation: (correlation.fraction_outside(), correlation.max_ratio()),
                lines: (line_test.fraction_outside(), line_test.mean_statistic() / line_test.expected_statistic()),
                fit,
            };
            let (relative, phase) = band_error(&row.fit.model, &plant, &antiresonance);
            println!(
                "{n:2} {m:2} {delay:2} | {bic:8.1} | {:7.1} % {:5.2} | {:7.1} % {:7.2} | {relative:12.4} | {phase:9.2} deg | {}{}{}",
                100.0 * row.correlation.0,
                row.correlation.1,
                100.0 * row.lines.0,
                row.lines.1,
                row.fit.iterations,
                if row.fit.converged { "" } else { " (not converged)" },
                if row.passes() { "" } else { " rejected" },
            );
            rows.push(row);
        }
    }

    // Selection: the lowest BIC among the candidates passing both tests
    let selected = rows.iter().filter(|r| r.passes()).min_by(|a, b| a.bic.total_cmp(&b.bic)).unwrap();
    println!("selected: n = {}, m = {}, nk = {}", selected.n, selected.m, selected.fit.model.delay);
    assert_eq!((selected.n, selected.m, selected.fit.model.delay), (5, 2, 8));

    let (relative, phase) = band_error(&selected.fit.model, &plant, &antiresonance);
    assert!(relative < 0.02 && phase < 1.0, "selected model: relative error {relative:e}, phase error {phase} deg");

    // The lower orders and the wrong delays are rejected by the cross-correlation test, the lower
    // orders also by the mean F of the line test
    for r in rows.iter().filter(|r| r.n < 5 || (r.m == 2 && r.fit.model.delay != 8)) {
        assert!(r.correlation.0 > 0.3 && r.correlation.1 > 2.0, "({}, {}, {}) not rejected by the cross-correlation", r.n, r.m, r.fit.model.delay);
    }
    for r in rows.iter().filter(|r| r.n < 5) {
        assert!(r.lines.1 > 2.0, "({}, {}, {}): mean F {}", r.n, r.m, r.fit.model.delay, r.lines.1);
    }

    // An extra zero passes the tests when converged, but loses in BIC
    let extra: Vec<&Row> = rows.iter().filter(|r| r.m == 3 && r.passes()).collect();
    assert!(!extra.is_empty() && extra.iter().all(|r| r.bic > selected.bic));
}

/// A constant input offset that is not in the recorded input (a torque bias in an open-loop test)
/// is integrated by the plant into a ramp of the output. SRIVC minimizes the output error, so the
/// ramp dominates the fit: an offset of 0.1 % of the input RMS gives a relative error of ~3 %
/// (0.35 % without it), and 1 % gives ~40 %.
/// The same high-pass filter (3 Hz, twice, below the excited band) on the input and the output
/// restores the estimate.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_input_offset() {
    let plant = two_inertia_plant();
    let (m, n) = (plant.numerator.len() - 1, plant.denominator.len() - 1);
    let w0 = 2.0 * PI * 60.0;
    let svf = 2.0 * PI * 100.0;
    let antiresonance = 36..=44;
    let filter = |x: &[f64]| high_pass(&high_pass(x, 3.0), 3.0);

    for offset in [0.0, 0.001, 0.01] {
        let (u, y) = experiment(&plant, 40000, 0x1234_5678_9abc, 0.05, offset);

        let raw = identify_normalized(&u, &y, m, n, plant.delay, svf, w0).unwrap();
        let (raw_error, raw_phase) = band_error(&raw.model, &plant, &antiresonance);

        let filtered = identify_normalized(&filter(&u), &filter(&y), m, n, plant.delay, svf, w0).unwrap();
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

/// Order 8 (three inertias, position output, low-pass): the plain call (`w0 = 1`) and the call in
/// normalized time both converge in a few iterations (4 without noise, 6 with), to an error of
/// ~1e-6 without noise, and agree. Before the prefilter of `identify` was scaled by the root radius
/// of `A(s)`, the plain call ran into the iteration limit here (error ~5e-4 without noise): the
/// matrix exponential of the unscaled companion matrix, whose entries span ~22 decades, was
/// inaccurate.
///
/// The position of a free body drifts in an open-loop test, so the signals are high-passed as in
/// `test_srivc_input_offset`.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
fn test_srivc_time_normalization() {
    let plant = three_inertia_plant();
    let (m, n) = (plant.numerator.len() - 1, plant.denominator.len() - 1);
    let svf = 2.0 * PI * 100.0;
    let nothing = 0..=0;
    let filter = |x: &[f64]| high_pass(&high_pass(x, 3.0), 3.0);

    for noise_ratio in [0.0, 0.05] {
        // Noise relative to the high-passed output (the raw position is dominated by its ramp)
        let (u, y0) = experiment(&plant, 40000, 0x1234_5678_9abc, 0.0, 0.0);
        let (u, y0) = (filter(&u), filter(&y0));
        let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
        let scale = noise_ratio * rms(&y0);
        let y: Vec<f64> = y0.iter().map(|v| v + scale * rng.gaussian()).collect();

        let mut errors = Vec::new();
        for (name, w0) in [("plain", 1.0), ("normalized", 2.0 * PI * 60.0)] {
            let fit = identify_normalized(&u, &y, m, n, plant.delay, svf, w0).unwrap();
            let (error, phase) = band_error(&fit.model, &plant, &nothing);
            println!(
                "noise {noise_ratio}: {name:10} {:2} iterations{}, relative error {error:.3e}, phase error {phase:.3} deg",
                fit.iterations,
                if fit.converged { "" } else { " (not converged)" },
            );
            let tolerance = if noise_ratio == 0.0 { 1e-4 } else { 5e-2 };
            assert!(fit.converged && fit.iterations <= 10 && error < tolerance, "noise {noise_ratio}: {name}, error {error:e}");
            errors.push(error);
        }
        // Same estimate up to rounding
        assert!((errors[0] - errors[1]).abs() < 1e-2 * errors[1].max(1e-6), "noise {noise_ratio}: {errors:?}");
    }
}
