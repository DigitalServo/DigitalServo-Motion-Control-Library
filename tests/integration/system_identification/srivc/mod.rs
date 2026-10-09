//! SRIVC: the basic identification and the dead time (`basic`), and on plants typical of an
//! open-loop motion-control identification test (the other submodules, with the settings shared
//! here): multi-inertia
//! mechanics (rigid-body integrator, lightly damped resonances / antiresonances) seen through an
//! analog low-pass filter and a dead time, excited by a band-limited multisine. Everything but
//! the synthetic data (plants, noise) and the comparison with the true plant is the library's:
//! `TransferFunctionWithDelay` (plant, simulation, identified model), `signal::excitation`,
//! `preprocessing::high_pass`, `srivc::search` with `validation::Check`.
//!
//! Six things are checked, each of them needed to use `srivc::identify` on such data:
//!
//! 1. `search::test_srivc_structure_search`: the denominator order `n`, the numerator order `m` and the
//!    input delay `nk` have to be searched *jointly*, and validated by the residual tests of
//!    `Validation` (not by the raw validation error, which differs only in the 4th digit between
//!    candidates), then compared by BIC (`srivc::search`).
//! 2. `input_offset::test_srivc_input_offset`: a small constant input offset (drift of the integrator) ruins the
//!    estimate; the same high-pass filter on the input and the output removes the problem.
//! 3. `high_order::test_srivc_high_order`: at high orders the coefficients of `A(s)` span many decades;
//!    `srivc::identify` scales its prefilter internally, so that it still converges.
//! 4. `pseudo_integration::test_srivc_pseudo_integration`: the pole of the rigid-body mode is not in band-limited,
//!    finite-length data; `srivc::identify_with_prefilter` identifies `s G(s)` (finite gain)
//!    through a pseudo-integrating prefilter instead, with no integrator applied to the data.
//! 5. `pseudo_integration::test_srivc_pseudo_integration_two_integrators`: the same with two integrators; the
//!    transient of an input offset is left out of the sums by `SrivcOptions::evaluated_from`.
//! 6. `search::test_srivc_integrator_search`: the number of integrators `q` is searched with `(n, m, nk)`
//!    and picked by BIC, a wrong `q` costing a parameter (a pole or a zero near the origin).
//!
//! The tests take a few seconds with optimizations and several minutes without, so they are
//! ignored in debug builds: run `cargo test --release --test integration srivc:: -- --nocapture`
//! (the tables of the candidates are printed), or force them with `-- --ignored`.

mod basic;
mod high_order;
mod input_offset;
mod pseudo_integration;
mod search;

use std::f64::consts::PI;

use dsmc::{Polynomial, TransferFunction, TransferFunctionWithDelay};
use dsmc::signal::excitation::multisine;

const TS: f64 = 1e-4;
/// Fundamental frequency of the multisine \[Hz\] (period 1 s).
const F0: f64 = 1.0;
/// Samples per period of the multisine (for the options of SRIVC, in samples).
const PERIOD: usize = 10_000;
/// Excited band of the multisine \[Hz\] (harmonics of the period).
const BAND: std::ops::RangeInclusive<usize> = 10..=200;

// ---------------------------------------------------------------------------------------------
// Synthetic data and comparison with the true plant
// ---------------------------------------------------------------------------------------------

/// xorshift64 pseudo-random numbers (measurement noise).
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

/// `s^2 + 2 ζ ω s + ω^2` with `ω = 2π f`.
fn second_order(f: f64, zeta: f64) -> Polynomial<f64> {
    let w = 2.0 * PI * f;
    Polynomial(vec![1.0, 2.0 * zeta * w, w * w])
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

/// Open-loop experiment from rest: multisine input `u` over `BAND` (`signal::excitation::multisine`,
/// equal amplitudes, unit RMS), and the output of `plant` to
/// `u + input_offset` plus white noise of standard deviation `noise_ratio` times the RMS of the
/// noise-free output *without the offset* (about its mean), so that the noise does not grow
/// with the drift of the offset (`~ t^2` for a position output). The offset is not part of the
/// recorded input.
fn experiment(plant: &TransferFunctionWithDelay<f64>, n_samples: usize, seed: u64, noise_ratio: f64, input_offset: f64) -> (Vec<f64>, Vec<f64>) {
    let harmonics: Vec<usize> = BAND.collect();
    let u = multisine(n_samples as f64 * TS, TS, F0, &harmonics, |_| 1.0, seed).unwrap();
    let applied: Vec<f64> = u.iter().map(|x| x + input_offset).collect();
    let y0 = plant.simulate(TS, &applied).unwrap();
    let without_offset = plant.simulate(TS, &u).unwrap();
    let mean = without_offset.iter().sum::<f64>() / n_samples as f64;
    let scale = noise_ratio * rms(&without_offset.iter().map(|v| v - mean).collect::<Vec<_>>());
    let mut rng = Xorshift(seed ^ 0x0abc_def1_2345);
    let y = y0.iter().map(|v| v + scale * rng.gaussian()).collect();
    (u, y)
}

/// Model error against the true plant over `BAND`, leaving out `exclude` \[Hz\] (the notch of an
/// antiresonance, where the relative error is dominated by the tiny gain):
/// (max relative error `|G - G0| / |G0|`, max phase error \[deg\]).
fn band_error(model: &TransferFunctionWithDelay<f64>, plant: &TransferFunctionWithDelay<f64>, exclude: &std::ops::RangeInclusive<usize>) -> (f64, f64) {
    BAND.filter(|f| !exclude.contains(f)).fold((0.0, 0.0), |(relative, phase): (f64, f64), f| {
        let omega = 2.0 * PI * f as f64;
        let (g, g0) = (model.frequency_response(omega), plant.frequency_response(omega));
        (relative.max((g - g0).norm() / g0.norm()), phase.max((g / g0).arg().abs().to_degrees()))
    })
}

/// `(m, n)` of a model.
fn orders(g: &TransferFunctionWithDelay<f64>) -> (usize, usize) {
    (g.tf.numerator.len() - 1, g.tf.denominator.len() - 1)
}

// ---------------------------------------------------------------------------------------------
// Plants
// ---------------------------------------------------------------------------------------------

/// Two-inertia system, torque -> velocity (antiresonance 40 Hz, resonance 65 Hz, ζ = 0.02),
/// second-order Butterworth low-pass at 300 Hz in the measurement, dead time of 8 samples (0.8 ms):
/// `n = 5`, `m = 2`, `nk = 8`.
fn two_inertia_plant() -> TransferFunctionWithDelay<f64> {
    let (antiresonance, resonance, low_pass) = (second_order(40.0, 0.02), second_order(65.0, 0.02), second_order(300.0, 0.5f64.sqrt()));
    let gain = 100.0;
    let gain_adjuster = resonance[2] / antiresonance[2] * low_pass[2];
    let numerator = &antiresonance * gain * gain_adjuster;
    let denominator = &(&Polynomial(vec![1.0, 0.0]) * &resonance) * &low_pass;
    TransferFunctionWithDelay::new(TransferFunction::from_polynomials(numerator, denominator), 8.0 * TS)
}

/// Three-inertia system, torque -> position (antiresonances 40 / 90 Hz, resonances 65 / 120 Hz),
/// the same low-pass and dead time: `n = 8`, `m = 4`, `nk = 8`.
fn three_inertia_plant() -> TransferFunctionWithDelay<f64> {
    let zeros = &second_order(40.0, 0.02) * &second_order(90.0, 0.02);
    let poles = &second_order(65.0, 0.02) * &second_order(120.0, 0.02);
    let low_pass = second_order(300.0, 0.5f64.sqrt());
    let dc_gain = 1e4;
    let gain_adjuster = poles[4] / zeros[4] * low_pass[2];
    let numerator = &zeros * dc_gain * gain_adjuster;
    let denominator = &(&Polynomial(vec![1.0, 0.0, 0.0]) * &poles) * &low_pass;
    TransferFunctionWithDelay::new(TransferFunction::from_polynomials(numerator, denominator), 8.0 * TS)
}
