//! `StateVariableFilter`: the derivatives of `x = v / A(s)` against the transfer functions
//! `s^i / A(s)` discretized by `Zoh` (held input), and the steady-state response to a sinusoid
//! (input linear between samples).

use std::f64::consts::PI;

use dsmc::discretize::{InterSample, StateVariableFilter, Zoh};
use dsmc::{tf, DiscreteSystem};

use super::TS;

/// Standard normal pseudo-random numbers (xorshift64, Box-Muller).
fn gaussian(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n).map(|_| (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos()).collect()
}

/// Held input: `λ^(N-i) x^(i)` is the response of `λ^(N-i) s^i / (s + λ)^N` discretized by `Zoh`
/// (exact for a held input, as the filter), sample by sample.
#[test]
fn test_state_variable_filter_zoh() {
    let (order, lambda) = (4, 2.0 * PI * 50.0);
    let u = gaussian(5000);
    let filter = StateVariableFilter::lag(order, lambda, TS);
    assert_eq!(filter.order(), order);
    let x = filter.apply(&u, InterSample::ZeroOrderHold);
    assert_eq!(x.shape(), (u.len(), order + 1));
    for i in 0..=order {
        let g = tf!("{} s^{i} / (s + {lambda})^{order}", lambda.powi((order - i) as i32));
        let mut system = DiscreteSystem::try_from(&g.discretize(Zoh, TS).unwrap()).unwrap();
        let expected: Vec<f64> = u.iter().map(|&v| system.update(v)).collect();
        let scale = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let error = (0..u.len()).fold(0.0f64, |m, k| m.max((lambda.powi((order - i) as i32) * x[(k, i)] - expected[k]).abs()));
        println!("s^{i}: max error {:.2e} of the max {scale:.3}", error / scale);
        assert!(error < 1e-9 * scale, "s^{i}: {error:e} of {scale}");
    }
}

/// Input linear between samples: in the steady state of a sinusoid `sin ω t`, the amplitude of
/// `x^(i)` is `|(jω)^i / (jω + λ)^N|`, within the error of the linear interpolation of the
/// sinusoid, `(ω ts)^2 / 12` relative (8e-5 here). The highest derivative
/// `x^(N) = v - Σ a_i x^(N-i)` is the difference of terms of the order of the input, so that
/// error is relative to the input there: against its own amplitude (`(ω / λ)^N` ~ 1e-4 at 5 Hz)
/// it is large, and it is checked against the input amplitude instead.
#[test]
fn test_state_variable_filter_foh_sinusoid() {
    let (order, lambda, f) = (4, 2.0 * PI * 50.0, 5.0);
    let w = 2.0 * PI * f;
    let period = (1.0 / (f * TS)).round() as usize;
    let v: Vec<f64> = (0..10 * period).map(|k| (w * k as f64 * TS).sin()).collect();
    let x = StateVariableFilter::lag(order, lambda, TS).apply_columns(&v, InterSample::FirstOrderHold);
    assert_eq!(x.len(), order + 1);
    // Amplitude over the last 5 periods (the transient decays with λ)
    let last = 5 * period;
    for (i, xi) in x.iter().enumerate() {
        let (mut a, mut b) = (0.0, 0.0);
        for (k, x) in xi.iter().enumerate().skip(v.len() - last) {
            let t = k as f64 * TS;
            a += x * (w * t).sin();
            b += x * (w * t).cos();
        }
        let amplitude = 2.0 / last as f64 * a.hypot(b);
        let expected = w.powi(i as i32) / (w * w + lambda * lambda).sqrt().powi(order as i32);
        println!("x^({i}): amplitude {amplitude:.6e}, expected {expected:.6e}");
        if i < order {
            assert!((amplitude / expected - 1.0).abs() < 1e-3, "x^({i}): {amplitude:e} vs {expected:e}");
        } else {
            assert!((amplitude - expected).abs() < (w * TS).powi(2), "x^({i}): {amplitude:e} vs {expected:e}");
        }
    }
}

/// `lag` is `new` with `a_k = C(N, k) λ^k`; `apply_columns` is `apply` by columns; `a = []` is the
/// identity.
#[test]
fn test_state_variable_filter_constructors() {
    let lambda = 20.0;
    let u = gaussian(200);
    let lag = StateVariableFilter::lag(3, lambda, TS).apply(&u, InterSample::FirstOrderHold);
    let new = StateVariableFilter::new(&[3.0 * lambda, 3.0 * lambda * lambda, lambda.powi(3)], TS).apply(&u, InterSample::FirstOrderHold);
    assert_eq!(lag, new);
    let columns = StateVariableFilter::lag(3, lambda, TS).apply_columns(&u, InterSample::FirstOrderHold);
    for (i, column) in columns.iter().enumerate() {
        assert_eq!(column.as_slice(), lag.column(i).as_slice());
    }
    let identity = StateVariableFilter::new(&[], TS).apply_columns(&u, InterSample::ZeroOrderHold);
    assert_eq!(identity, vec![u.clone()]);
    assert_eq!(StateVariableFilter::<f64>::lag(0, lambda, TS).order(), 0);
}

/// High order and low cutoff (`N = 5`, `λ` = 2 Hz, `ts` = 0.1 ms, `λ ts` ~ 1e-3): the step response
/// of `(λ / (s + λ))^5` settles at 1 by the filter; discretized as a transfer function, its 5
/// poles at `z = e^(-λ ts)` leave a polynomial whose coefficients lose the low-frequency response
/// to rounding.
#[test]
fn test_state_variable_filter_high_order_low_cutoff() {
    let (order, lambda, ts) = (5, 2.0 * PI * 2.0, 1e-4);
    let u = vec![1.0; 30000]; // 3 s, ~ 75 / λ
    let x = StateVariableFilter::lag(order, lambda, ts).apply_columns(&u, InterSample::ZeroOrderHold);
    let filtered = lambda.powi(order as i32) * x[0][u.len() - 1];
    let g = tf!("({lambda} / (s + {lambda}))^{order}");
    let mut system = DiscreteSystem::try_from(&g.discretize(Zoh, ts).unwrap()).unwrap();
    let by_tf = u.iter().fold(0.0, |_, &v| system.update(v));
    println!("step response at 3 s: filter {filtered:.9}, transfer function {by_tf:.9}");
    assert!((filtered - 1.0).abs() < 1e-6, "filter: {filtered}");
}
