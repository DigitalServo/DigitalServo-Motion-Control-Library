# dsmc: Digitalservo Motion Control Library

[![crates.io](https://img.shields.io/crates/v/dsmc.svg)](https://crates.io/crates/dsmc)
[![docs.rs](https://docs.rs/dsmc/badge.svg)](https://docs.rs/dsmc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

A Rust library for designing and analyzing motion control systems: transfer functions and state
space models, discretization, frequency analysis, trajectory generation, feedforward control
(perfect tracking control), signal processing and system identification.

## Features

| Area | What you get |
| --- | --- |
| **Systems** | `TransferFunction` in `s` or `z` (the domain is a type parameter, so they cannot be mixed up), the `tf!` macro, arithmetic, pole-zero cancellation, poles / zeros, partial fractions; `TransferFunctionWithDelay` (dead time: frequency response, exact simulation of a held input); `StateSpace` |
| **Discretization** | Zero-order hold (exact), bilinear (Tustin), backward difference, matched z-transform in both directions, each for transfer functions and state-space models, chosen by type or at run time (`DiscretizeMethod`); running a discrete-time `TransferFunction` or `StateSpace` sample by sample (`DiscreteSystem`: plant simulation, controllers, filters) |
| **Frequency analysis** | `FrequencyTransferFunction` (`ω -> G(jω)`, including dead time), Bode diagram, Nyquist plot, FFT, Welch's method |
| **Laplace transform** | Inverse Laplace transform, stable (non-causal) inverse of nonminimum-phase systems, piecewise-polynomial signals |
| **Trajectories** | Modified trapezoid / sine / constant velocity, cycloid, harmonic, smoothstep of any smoothness |
| **Feedforward** | Multirate perfect tracking control (PTC), with pre-actuation for nonminimum-phase plants |
| **Signal processing** | Pseudo-differentiator, delay; excitation signals (random-phase multisine with given amplitudes, e.g. by the inverse gain of a model for a flat output spectrum, started so that a double integrator does not drift; chirp) |
| **System identification** | ARX models and linear regressions by least squares or Kalman filter, instrumental variables (IV) for ARX models, SRIVC for continuous-time models, model validation (BIC / AIC, whiteness, residual-input cross-correlation and coherence, test at the excited lines of a periodic input, comparison with the nonparametric frequency response; output error or one-step prediction error; several tests at once with pass / fail), search of the SRIVC model structure (orders and delay) by validation, high-pass preprocessing against drifts, Levy / Sanathanan-Koerner, vector fitting, Gaussian process regression |
| **Logging** | CSV output of any `Serialize` value |

## Installation

```sh
cargo add dsmc
```

or in `Cargo.toml`:

```toml
[dependencies]
dsmc = "0.1"
```

## Examples

### Transfer functions

```rust
use dsmc::{tf, TransferFunction};

// From a string: the variable (`s` or `z`) selects continuous / discrete time at compile time.
let wn = 100.0;
let plant = tf!("{} / (s^2 + {} s + {})", wn * wn, 0.2 * wn, wn * wn);
let controller = tf!("(10 s + 100) / s");

// Arithmetic (common poles / zeros are cancelled)
let open_loop = &controller * &plant;
println!("{open_loop}");

// Closed loop under unity negative feedback: L / (1 + L), same as `&open_loop / &(1.0 + &open_loop)`
let closed_loop = open_loop.unity_feedback();

// From coefficients (descending order), or parsed at runtime with error handling
let g = TransferFunction::continuous(&[1.0], &[1.0, 3.0, 2.0]);
let h: TransferFunction<f64> = "1 / (s + 1)^2".parse().unwrap();

// Poles, zeros, partial fractions and the inverse Laplace transform
let poles = g.pz_map().poles;
println!("{:.3}", g.partial_fraction().time_domain()); // x(t) = exp(-1.000t) - exp(-2.000t)
let x = h.inverse_laplace();
assert!((x(1.0) - (-1.0f64).exp()).abs() < 1e-9);
```

### Discretization and simulation

```rust
use dsmc::{tf, DiscreteSystem, StateSpace};
use dsmc::discretize::{BackwardDifference, DiscretizeMethod, MatchedZ, Tustin, Zoh, ZerosAtInfinity};

let g = tf!("100 / (s + 100)");
let ts = 1e-3;

let g_zoh = g.discretize(Zoh, ts).unwrap();
let g_tustin = g.discretize(Tustin, ts).unwrap();
let g_backward = g.discretize(BackwardDifference, ts).unwrap();
let g_matched = g.discretize(MatchedZ(ZerosAtInfinity::MinusOne), ts).unwrap();
let ss_tustin = StateSpace::try_from(&g).unwrap().discretize(Tustin, ts).unwrap();

// The method chosen at run time
let method = DiscretizeMethod::MatchedZ(ZerosAtInfinity::KeepOneDelay);
let g_z = g.discretize(method, ts).unwrap();

// Step response, one sample at a time: a DiscreteSystem runs a discrete-time transfer
// function (its difference equation; single input and output: scalar in, scalar out)...
let mut system = DiscreteSystem::try_from(&g_zoh).unwrap();
let y: Vec<f64> = (0..100).map(|_| system.update(1.0)).collect();
// ...or a discrete-time state-space model (any number of inputs / outputs)
let mut system = DiscreteSystem::from(StateSpace::try_from(&g).unwrap().discretize(Zoh, ts).unwrap());
let y: Vec<f64> = (0..100).map(|_| system.update(&[1.0]).unwrap()[0]).collect();
```

### Bode diagram and Nyquist plot

```rust
use dsmc::{tf, BodeDiagramPlotter, FrequencyTransferFunction, NyquistPlotter, RadPerSec};
use num_complex::Complex;

let plant = tf!("1 / (s (0.01 s + 1))");
let controller = tf!("(100 s + 1000) / s");

// Any element can be put in series, e.g. a dead time of 1 ms
let delay = FrequencyTransferFunction::new(|w: f64| Complex::new(0.0, -w * 1e-3).exp());
let open_loop = &(&controller * &plant).frequency_transfer_function() * &delay;

// Gain [dB] and phase [rad] from 0.1 Hz to 1 kHz
let bode = BodeDiagramPlotter::<f64>::new(0.1, 1000.0, 0.1, true).plot(&open_loop);
// The same in rad/s
let bode_rad = BodeDiagramPlotter::<f64, RadPerSec>::new(1.0, 6000.0, 1.0, true).plot(&open_loop);
let nyquist = NyquistPlotter::<f64>::new(0.1, 1000.0, 0.1).plot(&open_loop);

// Single points
let c = open_loop.characteristics::<RadPerSec>(100.0, true);
println!("{:.1} dB, {:.1} deg", c.gain, c.phase.to_degrees());
```

Results can be written to CSV directly (one row per point: `frequency, gain, phase` or `omega, re, im`):

```rust,no_run
use dsmc::{tf, BodeDiagramPlotter};
use dsmc::logger::DataStorage;

let bode = BodeDiagramPlotter::<f64>::new(0.1, 1000.0, 0.1, true).plot(&tf!("1 / (s + 1)"));

let mut csv = DataStorage::new("./out/bode.csv").unwrap().set_header(["frequency", "gain", "phase"]);
for point in &bode {
    csv.add(point).unwrap();
}
csv.close().unwrap();
```

### Trajectory and perfect tracking control

```rust
use dsmc::{tf, StateSpace};
use dsmc::feedforward::ptc::LiftedDiscretizedSystem;
use dsmc::trajectory::{self, ModifiedSine, Trajectory};

// Normalized motion profiles: position, velocity and acceleration over x = 0..1
let profile = ModifiedSine.generate(1.0_f64, 1001);

// Feedforward input that makes the output of a nonminimum-phase plant follow a smooth move
// exactly at every frame (with pre-actuation before the move starts)
let ts = 1e-4;
let plant = tf!("(1 - 0.001 s) / (s (0.005 s + 1) (0.001 s + 1))");
let y_d = trajectory::smoothstep::piecewise(1.0, 0.05, 0.02, 4); // distance, duration, start, smoothness

let model = StateSpace::normalized_controllable_canonical(&plant).unwrap();
let lifted = LiftedDiscretizedSystem::new(&model, ts).unwrap();
let u = lifted.calculate_ptc_input_for_reference_output(&y_d, 900).unwrap();
```

### System identification

```rust
use dsmc::tf;
use dsmc::DiscreteSystem;
use dsmc::discretize::Zoh;
use dsmc::system_identification::lsm;

let ts = 1e-3;
let mut plant = DiscreteSystem::try_from(&tf!("1000 / (s^2 + 20 s + 1000)").discretize(Zoh, ts).unwrap()).unwrap();

// ARX model y[k] = a1 y[k-1] + a2 y[k-2] + b0 u[k-1] + b1 u[k-2]
let mut arx = lsm::arx::DataBuffer::<f64>::new(2, 1).with_input_delay(1);
let mut y_prev = 0.0;
for k in 0..1000 {
    let t = k as f64 * ts;
    let u = (1..50).map(|i| (i as f64 * 10.0 * t).sin()).sum::<f64>();
    let y = plant.update(u);
    arx.add(u, y_prev, y);
    y_prev = y;
}
let g_z = arx.identify().unwrap();
println!("{g_z}");
```

With measurement noise on the output, least squares is biased. The instrumental variable (IV)
method removes the bias, either for the discrete-time ARX model or, with SRIVC (simplified refined
instrumental variable method for continuous-time systems), directly for `G(s)` from sampled data:

```rust
use dsmc::tf;
use dsmc::DiscreteSystem;
use dsmc::discretize::Zoh;
use dsmc::system_identification::{arx::Arx, iv};
use dsmc::system_identification::iv::srivc::{Initialization, SrivcOptions};

let ts = 1e-3;
let mut plant = DiscreteSystem::try_from(&tf!("1000 / (s^2 + 20 s + 1000)").discretize(Zoh, ts).unwrap()).unwrap();
let u: Vec<f64> = (0..5000)
    .map(|k| (1..50).map(|i| (i as f64 * 10.0 * k as f64 * ts).sin()).sum::<f64>())
    .collect();
let y: Vec<f64> = u.iter().map(|&uk| plant.update(uk)).collect(); // + noise

// ARX model by least squares, then 3 IV iterations (each with the previous estimate as the
// auxiliary model generating the instruments)
let model = iv::arx::identify(&u, &y, &Arx::new(2, 1).with_input_delay(1), 3).unwrap();
let g_z = model.transfer_function();

// Continuous-time B(s) / A(s) with deg A = 2, deg B = 0 (orders: denominator first), started
// from a state-variable filter 1 / (s + 30)^2
let result = iv::srivc::identify(
    &u, &y, ts, 2, 0, &Initialization::StateVariableFilter(30.0), &SrivcOptions::default(),
).unwrap();
let g_s = result.model;
```

An identified model is validated on another experiment (`system_identification::validation`)
by its residual `ε = y - ŷ`, with `ŷ` the output of the model simulated with the measured input
(output error: `Validation::continuous`, `Validation::discrete`) or, for an ARX model with its
noise model, the one-step prediction (`Validation::one_step_prediction`):

| Method | What it checks | Input |
| --- | --- | --- |
| `bic`, `aic`, `aicc` | fit against the number of parameters, to compare model structures | any |
| `cross_correlation` | the residual does not depend on past inputs (wrong dynamics or delay) | any |
| `coherence_test` | the same, per frequency | any |
| `line_test` | the model error is at the noise level at every excited line | periodic |
| `autocorrelation` | the residual is white (Ljung-Box); for the output error, it tests the noise, not `G` | any |
| `frequency_response` | gain / phase error against the nonparametric estimate and its uncertainty (diagnostic) | any |

```rust
use dsmc::tf;
use dsmc::DiscreteSystem;
use dsmc::discretize::Zoh;
use dsmc::signal::excitation::chirp;
use dsmc::system_identification::validation::Validation;

let ts = 1e-3;
let u: Vec<f64> = chirp(20.0, ts, 0.5, 50.0).unwrap(); // 0.5 .. 50 Hz over 20 s (non-periodic)
let g_s = tf!("1000 / (s^2 + 20 s + 1000)");
let mut plant = DiscreteSystem::try_from(&g_s.discretize(Zoh, ts).unwrap()).unwrap();
let y: Vec<f64> = u.iter().map(|&uk| plant.update(uk)).collect(); // + noise

let validation = Validation::continuous(&g_s, ts, &u, &y).unwrap();
let (bic, aic) = (validation.bic(3), validation.aic(3)); // p = n + m + 1
let correlation = validation.cross_correlation(50, 2.58); // r(τ) within ±bound at 99 % per lag
let coherence = validation.coherence_test(1.0, 0.99).unwrap().excited(1e-2); // 1 s segments: 1 Hz bins
let response = validation.frequency_response(1.0).unwrap().excited(1e-2);
println!(
    "BIC {bic}, AIC {aic}, lags outside {}, bins outside {}, max gain error {} dB",
    correlation.fraction_outside(),
    coherence.fraction_outside(),
    response.gain_error_db().iter().fold(0.0f64, |m, e| m.max(e.abs())),
);
```

For an ARX model, the one-step prediction error must be white and independent of the input:

```rust
use dsmc::system_identification::{lsm, validation::Validation};

let noise = |k: usize| {
    // splitmix64: uniform in [-0.5, 0.5)
    let mut z = (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
};
let u: Vec<f64> = (0..2000).map(|k| noise(k + 10_000)).collect();
let mut y = vec![0.0; u.len()];
for k in 2..u.len() {
    let e = 0.01 * noise(k); // equation error
    y[k] = 1.5 * y[k - 1] - 0.7 * y[k - 2] + u[k - 1] + 0.5 * u[k - 2] + e;
}

let mut arx = lsm::arx::DataBuffer::<f64>::new(2, 1).with_input_delay(1);
for k in 0..u.len() {
    arx.add(u[k], if k > 0 { y[k - 1] } else { 0.0 }, y[k]);
}
arx.identify().unwrap();

let ts = 1e-3; // sampling period of the data
let validation = Validation::one_step_prediction(&arx.arx, ts, &u, &y).unwrap();
let whiteness = validation.autocorrelation(20, 2.58);
let white = whiteness.ljung_box <= whiteness.ljung_box_bound(2.33); // χ²(20) at 99 %
let independent = validation.cross_correlation(50, 2.58).fraction_outside() < 0.05;
let bic = validation.bic(arx.arx.parameter_len());
assert!(white && independent);
```

The tests can be run together with one confidence level by `Validation::check`, which also
decides pass / fail (whiteness by Ljung-Box, cross-correlation with a Bonferroni bound over the
lags, coherence and lines by the number outside against a binomial quantile) and prints a report:

```rust
use dsmc::tf;
use dsmc::system_identification::validation::{Check, CoherenceCheck, Validation};

let ts = 1e-3;
let g_s = tf!("1000 / (s^2 + 20 s + 1000)");
let noise = |k: usize| {
    // splitmix64: uniform in [-0.5, 0.5)
    let mut z = (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
};
let u: Vec<f64> = (0..10000).map(|k| noise(k + 100_000)).collect();
let y0 = Validation::continuous(&g_s, ts, &u, &u).unwrap().simulated; // plant output
let y: Vec<f64> = y0.iter().enumerate().map(|(k, v)| v + 0.01 * noise(k)).collect();

let report = Validation::continuous(&g_s, ts, &u, &y)
    .unwrap()
    .check(
        &[
            Check::InformationCriteria { parameters: 3 },
            Check::Whiteness { max_lag: 20 },
            Check::CrossCorrelation { max_lag: 50 },
            Check::Coherence(CoherenceCheck::new(1.0, 1e-2)),
        ],
        0.99,
    )
    .unwrap();
println!("{report}"); // one line per check, then "overall: passed"
assert!(report.passed());
```

A model that leaves out modes on purpose (e.g. a rigid-body model of a plant whose resonances
are above the control bandwidth) fails the tests on the whole band, rightly: the residual keeps
the error of those modes. `CoherenceCheck::new(segment_s, excited).set_band((low, high))` tests the
coherence only in `[low, high]` \[Hz\] (`CoherenceTest::band`, `FrequencyResponseComparison::band`
for a closer look), with a margin to the lowest mode left out; run it without the whiteness and
cross-correlation tests, which see the whole band.

Durations (`evaluated_from`, the segments of `coherence_test` and `frequency_response`) are in
seconds and the fundamental frequency of `line_test` in Hz, whole numbers of the sampling period
(the period `1 / f0` for the fundamental); the lags of `autocorrelation` and `cross_correlation`
are in samples.

With a periodic input (e.g. a multisine of fundamental frequency `f0` \[Hz\], excited at the
harmonics `lines`), `validation.evaluated_from(1.0 / f0)?.line_test(f0, &lines, 0.99)` compares
the model error at every excited line with the noise level estimated from the period-to-period
variation (`Check::Lines { fundamental_frequency: f0, lines }` in `check`).

The structure of a continuous-time model (orders `n`, `m`, integrators `q` and input delay `nk`,
searched together: missing poles are made up for by a longer delay) is chosen by `srivc::search`: every candidate is
identified, validated on another experiment by `check`, and the lowest BIC among the candidates
passing every test is selected:

```rust
use dsmc::{tf, TransferFunctionWithDelay};
use dsmc::signal::excitation::multisine;
use dsmc::system_identification::iv::srivc::{self, Initialization, SearchOptions, Structure};
use dsmc::system_identification::validation::Check;

let ts = 1e-3;
let f0 = 1.0; // multisine of fundamental 1 Hz (period 1 s) over 1 .. 40 Hz
let lines: Vec<usize> = (1..=40).collect();
let plant = TransferFunctionWithDelay::new(tf!("1000 / (s^2 + 20 s + 1000)"), 3.0 * ts);
let noise = |k: u64| {
    // splitmix64: uniform in [-0.5, 0.5)
    let mut z = k.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
};
let experiment = |periods: f64, seed: u64| {
    let u: Vec<f64> = multisine(periods / f0, ts, f0, &lines, |_| 1.0, seed).unwrap();
    let y = plant.simulate(ts, &u).unwrap(); // plant output, plus measurement noise:
    let y: Vec<f64> = y.iter().enumerate().map(|(k, v)| v + 0.02 * noise(k as u64 + (seed << 32))).collect();
    (u, y)
};
let (u, y) = experiment(3.0, 1);
let (u_val, y_val) = experiment(5.0, 2);

// n = 1 ..= 3, m = 0 ..= 1, q = 0 (no pole at the origin), nk = 2 ..= 4
let structures = Structure::grid(1..=3, 0..=1, 0..=0, 2..=4);
let options = SearchOptions {
    evaluated_from: 1000, // samples: leave out the transient of the first period
    ..SearchOptions::new(
        Initialization::StateVariableFilter(30.0),
        vec![Check::CrossCorrelation { max_lag: 50 }, Check::Lines { fundamental_frequency: f0, lines: lines.clone() }],
        0.99,
    )
};
let search = srivc::search((&u, &y), (&u_val, &y_val), ts, &structures, &options).unwrap();
println!("{search}"); // one row per candidate: BIC, tests, iterations
assert_eq!(search.selected().unwrap().structure, Structure::new(2, 0, 0, 3));
```

A plant with a rigid-body mode (poles at, or very close to, the origin) is identified as
`R(s) / s^q` by `srivc::identify_with_prefilter` with a `Prefilter::new(q, omega_c)`: the input
is pseudo-integrated, `s / (s + ω_c)^(q+1)`, and the output high-passed, `s^(q+1) / (s + ω_c)^(q+1)`,
so that no integrator is applied to the data. With `SearchOptions::prefilter = Some(omega_c)`,
`srivc::search` searches `q` with the other parameters of the structure. `R` may be a constant
(`n = 0`, only with `q >= 1`): the pure integrator `b_0 / s^q`, e.g. a single inertia `1 / (J s^2)`
(torque -> position) is `(n, m, q) = (0, 0, 2)` with `b_0 = 1 / J`, or the rigid-body mode alone
of a plant whose resonances are above the band of interest.

Data with a drift (e.g. an integrating plant driven by an unknown input offset) are high-passed
before the identification by `preprocessing::high_pass(&u, &y, cutoff, ts, order)`, which filters
the input and the output alike from rest, so that they are still related by the same `G(s)`.

From measured frequency responses (e.g. `dsmc::fft::welch`), a continuous-time model can be fitted:

```rust
use dsmc::{tf, FrequencyResponse, TransferFunction};
use dsmc::system_identification::frequency_response::vector_fitting::{identify, VectorFittingOptions};

let g = tf!("1000 / (s^2 + 20 s + 1000)").frequency_transfer_function();
let samples: Vec<FrequencyResponse<f64>> = (1..=500)
    .map(|k| {
        let omega = 0.5 * k as f64;
        FrequencyResponse { omega, value: g.response(omega) }
    })
    .collect();

let fitted = identify(&samples, 2, &VectorFittingOptions::default()).unwrap();
let model: TransferFunction<f64> = fitted.into();
```

More examples are in [`tests/`](tests) and in the [API documentation](https://docs.rs/dsmc).

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, copy, modify,
merge, publish, distribute, sublicense and sell copies of the software, provided that the
copyright notice and the permission notice are included in all copies or substantial portions
of the software. The software is provided "as is", without warranty of any kind.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you shall be licensed under the MIT License, without any additional terms or
conditions.
