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
| **Systems** | `TransferFunction` in `s` or `z` (the domain is a type parameter, so they cannot be mixed up), the `tf!` macro, arithmetic, pole-zero cancellation, poles / zeros, partial fractions; `StateSpace` |
| **Discretization** | Zero-order hold (exact), bilinear (Tustin), matched z-transform in both directions, sample-by-sample simulators |
| **Frequency analysis** | `FrequencyTransferFunction` (`ω -> G(jω)`, including dead time), Bode diagram, Nyquist plot, FFT, Welch's method |
| **Laplace transform** | Inverse Laplace transform, stable (non-causal) inverse of nonminimum-phase systems, piecewise-polynomial signals |
| **Trajectories** | Modified trapezoid / sine / constant velocity, cycloid, harmonic, smoothstep of any smoothness |
| **Feedforward** | Multirate perfect tracking control (PTC), with pre-actuation for nonminimum-phase plants |
| **Signal processing** | Pseudo-differentiator, delay |
| **System identification** | ARX models and linear regressions by least squares or Kalman filter, instrumental variables (IV) for ARX models, SRIVC for continuous-time models, model validation (BIC, residual-input cross-correlation, test at the excited lines of a periodic input), Levy / Sanathanan-Koerner, vector fitting, Gaussian process regression |
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
use dsmc::tf;
use dsmc::discretize::{bilinear_transform, exact_discretize, matched_z_transform};
use dsmc::discretize::exact_discretize::DiscretizedSystem;
use dsmc::discretize::matched_z_transform::ZerosAtInfinity;

let g = tf!("100 / (s + 100)");
let ts = 1e-3;

let g_zoh = exact_discretize::discretize(&g, ts).unwrap();
let g_tustin = bilinear_transform::discretize(&g, ts);
let g_matched = matched_z_transform::to_discrete(&g, ts, ZerosAtInfinity::MinusOne).unwrap();

// Step response, one sample at a time
let mut system = DiscretizedSystem::from_tf(&g, ts).unwrap();
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

let mut csv = DataStorage::new("./out/bode.csv", ',', true).unwrap();
for point in &bode {
    csv.add(point).unwrap();
}
csv.close().unwrap();
```

### Trajectory and perfect tracking control

```rust
use dsmc::tf;
use dsmc::discretize::exact_discretize::DiscretizedSystem;
use dsmc::feedforward::ptc::LiftedDiscretizedSystem;
use dsmc::trajectory::{self, ModifiedSine, Trajectory};

// Normalized motion profiles: position, velocity and acceleration over x = 0..1
let profile = ModifiedSine.generate(1.0_f64, 1001);

// Feedforward input that makes the output of a nonminimum-phase plant follow a smooth move
// exactly at every frame (with pre-actuation before the move starts)
let ts = 1e-4;
let plant = tf!("(1 - 0.001 s) / (s (0.005 s + 1) (0.001 s + 1))");
let y_d = trajectory::smoothstep::piecewise(1.0, 0.05, 0.02, 4); // distance, duration, start, smoothness

let model = DiscretizedSystem::from_tf_normalized(&plant, ts).unwrap();
let lifted: LiftedDiscretizedSystem<f64> = model.try_into().unwrap();
let u = lifted.calculate_ptc_input_for_reference_output(&y_d, 900).unwrap();
```

### System identification

```rust
use dsmc::tf;
use dsmc::discretize::exact_discretize::DiscretizedSystem;
use dsmc::system_identification::lsm;

let ts = 1e-3;
let mut plant = DiscretizedSystem::from_tf(&tf!("1000 / (s^2 + 20 s + 1000)"), ts).unwrap();

// ARX model y[k] = a1 y[k-1] + a2 y[k-2] + b0 u[k-1] + b1 u[k-2]
let mut arx = lsm::arx::DataBuffer::<f64>::new(1, 2).with_input_delay(1);
for k in 0..1000 {
    let t = k as f64 * ts;
    let u = (1..50).map(|i| (i as f64 * 10.0 * t).sin()).sum::<f64>();
    let y_prev = plant.output[0];
    let y = plant.update(&[u]).unwrap()[0];
    arx.add(u, y_prev, y);
}
let g_z = arx.identify().unwrap();
println!("{g_z}");
```

With measurement noise on the output, least squares is biased. The instrumental variable (IV)
method removes the bias, either for the discrete-time ARX model or, with SRIVC (simplified refined
instrumental variable method for continuous-time systems), directly for `G(s)` from sampled data:

```rust
use dsmc::tf;
use dsmc::discretize::exact_discretize::DiscretizedSystem;
use dsmc::system_identification::{arx::Arx, iv};
use dsmc::system_identification::iv::srivc::{Initialization, SrivcOptions};

let ts = 1e-3;
let mut plant = DiscretizedSystem::from_tf(&tf!("1000 / (s^2 + 20 s + 1000)"), ts).unwrap();
let u: Vec<f64> = (0..5000)
    .map(|k| (1..50).map(|i| (i as f64 * 10.0 * k as f64 * ts).sin()).sum::<f64>())
    .collect();
let y: Vec<f64> = u.iter().map(|&uk| plant.update(&[uk]).unwrap()[0]).collect(); // + noise

// ARX model by least squares, then 3 IV iterations (each with the previous estimate as the
// auxiliary model generating the instruments)
let model = iv::arx::identify(&u, &y, &Arx::new(1, 2).with_input_delay(1), 3).unwrap();
let g_z = model.transfer_function();

// Continuous-time B(s) / A(s) with deg B = 0, deg A = 2, started from a state-variable filter
// 1 / (s + 30)^2
let result = iv::srivc::identify(
    &u, &y, ts, 0, 2, &Initialization::StateVariableFilter(30.0), &SrivcOptions::default(),
).unwrap();
let g_s = result.model;
```

An identified model is validated on another experiment by simulating it with the measured input:
BIC for comparing model structures, and tests of the residual (it must not depend on the input,
and, with a periodic input, must be at the noise level at every excited line):

```rust
use dsmc::tf;
use dsmc::discretize::exact_discretize::DiscretizedSystem;
use dsmc::system_identification::validation::Validation;

let ts = 1e-3;
let period = 1000; // multisine of period 1 s, lines at 1 .. 40 Hz
let lines: Vec<usize> = (1..=40).collect();
let u: Vec<f64> = (0..6 * period)
    .map(|k| lines.iter().map(|&l| (2.0 * std::f64::consts::PI * (l * k) as f64 / period as f64 + l as f64).sin()).sum())
    .collect();
let g_s = tf!("1000 / (s^2 + 20 s + 1000)");
let mut plant = DiscretizedSystem::from_tf(&g_s, ts).unwrap();
let y: Vec<f64> = u.iter().map(|&uk| plant.update(&[uk]).unwrap()[0]).collect(); // + noise

let validation = Validation::continuous(&g_s, 0, ts, &u, &y).unwrap();
let bic = validation.bic(3); // N ln V + p ln N, p = n + m + 1
let correlation = validation.cross_correlation(50, 2.58); // r(τ) within ±bound at 99 % per lag
let line_test = validation.evaluated_from(period).line_test(period, &lines, 0.99).unwrap();
println!("BIC {bic}, lags outside {}, lines outside {}", correlation.fraction_outside(), line_test.fraction_outside());
```

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

MIT
