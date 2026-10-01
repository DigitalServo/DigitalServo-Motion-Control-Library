use dsmc::{tf, Polynomial, TransferFunction};

fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!((actual - expected).abs() <= tol, "{actual} != {expected}");
}

/// Compare `x(t)` from the expansion with its analytic inverse Laplace transform.
fn check(x: &TransferFunction<f64>, analytic: impl Fn(f64) -> f64) {
    let pf = x.partial_fraction();
    for i in 0..=50 {
        let t = 0.1 * i as f64;
        assert_close(pf.time_response(t), analytic(t), 1e-6);
    }
    assert_eq!(pf.time_response(-1.0), 0.0);
}

#[test]
fn distinct_real_poles() {
    // 1 / ((s+1)(s+2)) = 1/(s+1) - 1/(s+2)
    let x = tf!("1 / ((s + 1)(s + 2))");
    check(&x, |t| (-t).exp() - (-2.0 * t).exp());
    println!("{}\n{}", x.partial_fraction(), x.partial_fraction().time_domain());
}

#[test]
fn repeated_poles() {
    // 1 / (s (s+1)^3) = 1/s - 1/(s+1) - 1/(s+1)^2 - 1/(s+1)^3
    let x = tf!("1 / (s (s + 1)^3)");
    let pf = x.partial_fraction();
    let triple = pf.terms.iter().find(|p| p.multiplicity() == 3).unwrap();
    assert_close(triple.pole.re, -1.0, 1e-12);
    for r in &triple.residues {
        assert_close(r.re, -1.0, 1e-12);
    }
    check(&x, |t| 1.0 - (-t).exp() * (1.0 + t + t * t / 2.0));
    println!("{:.2}\n{:.2}", pf, pf.time_domain());
}

#[test]
fn complex_poles() {
    // (s + 3) / ((s+1)^2 + 4) = e^{-t} (cos 2t + sin 2t)
    let x = tf!("(s + 3) / (s^2 + 2s + 5)");
    check(&x, |t| (-t).exp() * ((2.0 * t).cos() + (2.0 * t).sin()));
    println!("{:.3}", x.partial_fraction().time_domain());
}

#[test]
fn repeated_complex_poles() {
    // 2ω^3/(s^2+ω^2)^2 = sin(ωt) - ωt cos(ωt), ω = 1
    let x = tf!("2 / (s^2 + 1)^2");
    check(&x, |t| t.sin() - t * t.cos());
    println!("{:.3}", x.partial_fraction().time_domain());
}

#[test]
fn improper() {
    // (s^2 + 3s + 3) / (s + 1) = s + 2 + 1/(s+1)
    let x = tf!("(s^2 + 3s + 3) / (s + 1)");
    let pf = x.partial_fraction();
    assert_eq!(pf.direct.len(), 2);
    assert_close(pf.direct[0], 1.0, 1e-9);
    assert_close(pf.direct[1], 2.0, 1e-9);
    check(&x, |t| (-t).exp());
    println!("{}\n{}", pf, pf.time_domain());
}

#[test]
fn repeated_poles_exact_coefficients() {
    // Built without `reduced()`, so the coefficients of s (s+1)^3 are exact.
    let x: TransferFunction<f64> =
        TransferFunction::from_polynomials(Polynomial(vec![1.0]), Polynomial(vec![1.0, 3.0, 3.0, 1.0, 0.0]));
    let pf = x.partial_fraction();
    let triple = pf.terms.iter().find(|p| p.multiplicity() == 3).unwrap();
    assert_close(triple.pole.re, -1.0, 1e-12);
    for r in &triple.residues {
        assert_close(r.re, -1.0, 1e-12);
    }
}

fn assert_coeffs(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tol, "{actual:?} != {expected:?}");
    }
}

#[test]
fn reduced_keeps_remaining_repeated_root() {
    // (s+1) / ((s+1)(s+2)^3) -> 1 / (s+2)^3 = 1 / (s^3 + 6s^2 + 12s + 8)
    let x = tf!("(s + 1) / ((s + 1)(s + 2)^3)");
    assert_coeffs(&x.numerator, &[1.0], 1e-12);
    assert_coeffs(&x.denominator, &[1.0, 6.0, 12.0, 8.0], 1e-12);
    check(&x, |t| t * t / 2.0 * (-2.0 * t).exp());
}

#[test]
fn reduced_repeated_and_complex_common_factors() {
    // (s+1)^2 (s^2+2s+5) / ((s+1)^3 (s^2+2s+5) (s+3)) -> 1 / ((s+1)(s+3))
    let x = tf!("(s + 1)^2 (s^2 + 2s + 5) / ((s + 1)^3 (s^2 + 2s + 5) (s + 3))");
    assert_coeffs(&x.numerator, &[1.0], 1e-9);
    assert_coeffs(&x.denominator, &[1.0, 4.0, 3.0], 1e-9);

    // (s+1)^3 / (s+1)^2 -> s + 1
    let y = tf!("(s + 1)^3 / (s + 1)^2");
    assert_coeffs(&y.numerator, &[1.0, 1.0], 1e-9);
    assert_coeffs(&y.denominator, &[1.0], 1e-9);
}

#[test]
fn time_function_closure() {
    // The closure owns its data, so it outlives the transfer function it came from.
    let x = {
        let g = tf!("(s + 3) / (s^2 + 2s + 5)");
        g.inverse_laplace()
    };
    let y = tf!("1 / ((s + 1)(s + 2))").partial_fraction().time_function();
    for i in 0..=50 {
        let t = 0.1 * i as f64;
        assert_close(x(t), (-t).exp() * ((2.0 * t).cos() + (2.0 * t).sin()), 1e-9);
        assert_close(y(t), (-t).exp() - (-2.0 * t).exp(), 1e-9);
    }
    assert_eq!(x(-1.0), 0.0);

    // Usable wherever `Fn(f64) -> f64` is expected.
    let samples: Vec<f64> = (0..5).map(|i| i as f64 * 0.1).map(&y).collect();
    assert_eq!(samples.len(), 5);
}
