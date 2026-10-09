use dsmc::laplace_transform::{TimeDomain, TimeExpression, TimeMode, TrigForm};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, SQRT_2};
use dsmc::tf;

fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!((actual - expected).abs() <= tol, "{actual} != {expected}");
}

fn assert_coeffs(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} != {expected:?}");
    for (&a, &e) in actual.iter().zip(expected) {
        assert_close(a, e, tol);
    }
}

fn assert_mode(m: &TimeMode<f64>, sigma: f64, omega: f64, cos: &[f64], sin: &[f64]) {
    assert_close(m.sigma, sigma, 1e-9);
    assert_close(m.omega, omega, 1e-9);
    assert_coeffs(&m.cos, cos, 1e-9);
    assert_coeffs(&m.sin, sin, 1e-9);
}

/// The mode with the given `(σ, ω)`.
fn mode(expr: &TimeExpression<f64>, sigma: f64, omega: f64) -> &TimeMode<f64> {
    expr.modes
        .iter()
        .find(|m| (m.sigma - sigma).abs() < 1e-6 && (m.omega - omega).abs() < 1e-6)
        .unwrap_or_else(|| panic!("no mode ({sigma}, {omega}) in {expr:?}"))
}

#[test]
fn conjugate_pair_is_one_mode() {
    // (s + 3) / ((s+1)^2 + 4) = e^{-t} (cos 2t + sin 2t)
    let expr = tf!("(s + 3) / (s^2 + 2s + 5)").partial_fraction().time_expression();
    assert!(expr.impulses.is_empty());
    assert_eq!(expr.modes.len(), 1);
    assert_mode(&expr.modes[0], -1.0, 2.0, &[1.0], &[1.0]);
    assert!(expr.modes[0].is_oscillatory());
}

#[test]
fn repeated_real_pole() {
    // 1 / (s (s+1)^3) = 1 - e^{-t} (1 + t + t^2/2)
    let expr = tf!("1 / (s (s + 1)^3)").partial_fraction().time_expression();
    assert_eq!(expr.modes.len(), 2);
    assert_mode(mode(&expr, 0.0, 0.0), 0.0, 0.0, &[1.0], &[]);
    let m = mode(&expr, -1.0, 0.0);
    assert_mode(m, -1.0, 0.0, &[-1.0, -1.0, -0.5], &[]);
    assert_eq!(m.degree(), 2);
    assert!(!m.is_oscillatory());
}

#[test]
fn repeated_complex_pole() {
    // 2 / (s^2+1)^2 = sin t - t cos t
    let expr = tf!("2 / (s^2 + 1)^2").partial_fraction().time_expression();
    assert_eq!(expr.modes.len(), 1);
    assert_mode(&expr.modes[0], 0.0, 1.0, &[0.0, -1.0], &[1.0]);
}

#[test]
fn pure_cosine_keeps_frequency() {
    // s / (s^2 + 4) = cos 2t: no sin term, but still oscillatory
    let expr = tf!("s / (s^2 + 4)").partial_fraction().time_expression();
    assert_eq!(expr.modes.len(), 1);
    assert_mode(&expr.modes[0], 0.0, 2.0, &[1.0], &[]);
    for i in 0..=50 {
        let t = 0.1 * i as f64;
        assert_close(expr.eval(t), (2.0 * t).cos(), 1e-12);
    }
    assert_eq!(format!("{:.1}", expr), "(cos(2.0t))");
}

#[test]
fn impulses_ascending() {
    // (s^3 + 3s^2 + 3s + 2) / (s + 1) = s^2 + 2s + 1 + 1/(s+1)
    let expr = tf!("(s^3 + 3s^2 + 3s + 2) / (s + 1)").partial_fraction().time_expression();
    assert_coeffs(&expr.impulses, &[1.0, 2.0, 1.0], 1e-9);
    assert_eq!(format!("{:.1}", expr), "δ^(2)(t) + 2.0 * δ^(1)(t) + δ(t) + exp(-1.0t)");
}

#[test]
fn eval_matches_time_response() {
    for x in [
        tf!("1 / ((s + 1)(s + 2))"),
        tf!("(s + 3) / (s^2 + 2s + 5)"),
        tf!("2 / (s^2 + 1)^2"),
        tf!("(s - 1)(s^2 + 0.4s + 9) / ((s + 0.5)^2 (s^2 + s + 4)^2 (s + 3))"),
        tf!("1 / ((s + 1)(s + 1.001))"),
    ] {
        let pf = x.partial_fraction();
        let expr = pf.time_expression();
        let f = pf.time_function();
        for i in 0..=100 {
            let t = 0.1 * i as f64;
            assert_close(expr.eval(t), pf.time_response(t), 1e-9);
            assert_close(f(t), pf.time_response(t), 1e-9);
        }
        assert_eq!(f(-1.0), 0.0);
    }
}

#[test]
fn negation() {
    let expr = tf!("(s + 3) / (s^2 + 2s + 5)").partial_fraction().time_expression();
    let neg = -expr.clone();
    for i in 0..=20 {
        let t = 0.25 * i as f64;
        assert_close(neg.eval(t), -expr.eval(t), 1e-15);
    }
}

#[test]
fn amplitude_phase() {
    // e^{-t} (cos 2t + sin 2t) = √2 e^{-t} cos(2t - π/4) = √2 e^{-t} sin(2t + π/4)
    let expr = tf!("(s + 3) / (s^2 + 2s + 5)").partial_fraction().time_expression();
    let m = &expr.modes[0];
    assert_eq!(m.amplitude_phase(0, TrigForm::CosSin), None);
    let (a, phi) = m.amplitude_phase(0, TrigForm::Cos).unwrap();
    assert_close(a, SQRT_2, 1e-9);
    assert_close(phi, -FRAC_PI_4, 1e-9);
    let (a, phi) = m.amplitude_phase(0, TrigForm::Sin).unwrap();
    assert_close(a, SQRT_2, 1e-9);
    assert_close(phi, FRAC_PI_4, 1e-9);
    let display = |form| format!("{:.3}", TimeDomain::from(expr.clone()).trig_form(form));
    assert_eq!(display(TrigForm::Cos), "x(t) = 1.414 * exp(-1.000t) * cos(2.000t - 0.785)");
    assert_eq!(display(TrigForm::Sin), "x(t) = 1.414 * exp(-1.000t) * sin(2.000t + 0.785)");
    assert_eq!(display(TrigForm::CosSin), format!("x(t) = {:.3}", expr));
}

#[test]
fn amplitude_phase_of_powers_of_t() {
    // 2 / (s^2+1)^2 = sin t - t cos t = sin(t) + t sin(t - π/2)
    let expr = tf!("2 / (s^2 + 1)^2").partial_fraction().time_expression();
    let m = &expr.modes[0];
    let (a0, phi0) = m.amplitude_phase(0, TrigForm::Sin).unwrap();
    let (a1, phi1) = m.amplitude_phase(1, TrigForm::Sin).unwrap();
    assert_close(a0, 1.0, 1e-9);
    assert_close(phi0, 0.0, 1e-9);
    assert_close(a1, 1.0, 1e-9);
    assert_close(phi1, -FRAC_PI_2, 1e-9);
    println!("{:.3}", tf!("2 / (s^2 + 1)^2").partial_fraction().time_domain().trig_form(TrigForm::Sin));
}

/// `Σ_k A_k t^k e^{σt} cos(ωt + φ_k)` (or sin) reproduces `eval`.
#[test]
fn amplitude_phase_reconstructs_signal() {
    let x = tf!("(s - 1)(s^2 + 0.4s + 9) / ((s + 0.5)^2 (s^2 + s + 4)^2 (s + 3))");
    let expr = x.partial_fraction().time_expression();
    for form in [TrigForm::Cos, TrigForm::Sin] {
        for i in 0..=100 {
            let t = 0.1 * i as f64;
            let rebuilt: f64 = expr
                .modes
                .iter()
                .map(|m| {
                    let envelope = (m.sigma * t).exp();
                    (0..=m.degree())
                        .map(|k| {
                            let tk = t.powi(k as i32);
                            if !m.is_oscillatory() {
                                return m.cos.get(k).copied().unwrap_or(0.0) * tk * envelope;
                            }
                            let (a, phi) = m.amplitude_phase(k, form).unwrap();
                            let arg = m.omega * t + phi;
                            let trig = if form == TrigForm::Cos { arg.cos() } else { arg.sin() };
                            a * tk * envelope * trig
                        })
                        .sum::<f64>()
                })
                .sum();
            assert_close(rebuilt, expr.eval(t), 1e-12);
        }
    }
}
