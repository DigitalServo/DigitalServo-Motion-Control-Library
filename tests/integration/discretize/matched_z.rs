//! Matched z-transform and its inverse `to_continuous`.

use dsmc::discretize::matched_z_transform::{
    to_continuous, to_continuous_with, to_continuous_with_delay, MatchedZ,
    MatchedZError, ToContinuousOptions, ZerosAtInfinity,
};
use dsmc::{tf, Continuous, Discrete, Polynomial, TransferFunction};
use num_complex::Complex;

use super::{assert_coeffs_close, assert_same_tf, normalized, plants, TS};

#[test]
fn first_order() {
    let a: f64 = 100.0;
    let p = (-a * TS).exp();

    // All zeros at infinity to z = -1: K (z + 1) / (z - p), DC gain 1 -> K = (1 - p) / 2
    let g = tf!("100 / (s + 100)").discretize(MatchedZ(ZerosAtInfinity::MinusOne), TS).unwrap();
    assert_same_tf(&g, &TransferFunction::discrete(&[(1.0 - p) / 2.0, (1.0 - p) / 2.0], &[1.0, -p]), 1e-12, "MinusOne");

    // One kept at infinity: (1 - p) / (z - p)
    let g = tf!("100 / (s + 100)").discretize(MatchedZ(ZerosAtInfinity::KeepOneDelay), TS).unwrap();
    assert_same_tf(&g, &TransferFunction::discrete(&[1.0 - p], &[1.0, -p]), 1e-12, "KeepOneDelay");
}

#[test]
fn round_trip_s_z_s() {
    for g in plants() {
        for option in [ZerosAtInfinity::MinusOne, ZerosAtInfinity::KeepOneDelay] {
            let back = to_continuous(g.discretize(MatchedZ(option), TS).unwrap(), TS).unwrap();
            assert_same_tf(&back, &g, 1e-7, &format!("s -> z -> s, {option:?}"));
        }
    }
}

#[test]
fn round_trip_z_s_z() {
    for g in plants() {
        for option in [ZerosAtInfinity::MinusOne, ZerosAtInfinity::KeepOneDelay] {
            let gz = g.discretize(MatchedZ(option), TS).unwrap();
            let again = to_continuous(&gz, TS).unwrap().discretize(MatchedZ(option), TS).unwrap();
            assert_same_tf(&again, &gz, 1e-7, &format!("z -> s -> z, {option:?}"));
        }
    }
}

/// Well below the Nyquist frequency, G(e^(jωT)) ≈ G(jω).
#[test]
fn low_frequency_response() {
    for g in plants() {
        let gz = g.discretize(MatchedZ(ZerosAtInfinity::MinusOne), TS).unwrap();
        for w in [0.1, 1.0, 10.0] {
            let (hs, hz) = (g.frequency_transfer_function().response(w), gz.frequency_transfer_function(TS).response(w));
            assert!((hs - hz).norm() <= 2e-2 * hs.norm(), "{g:?} at {w}: {hs} vs {hz}");
        }
    }
}

#[test]
fn errors() {
    // Pure delay 1/z: pole at z = 0
    let delay = TransferFunction::<f64, Discrete>::from_polynomials(Polynomial(vec![1.0]), Polynomial(vec![1.0, 0.0]));
    assert_eq!(to_continuous(&delay, TS).unwrap_err(), MatchedZError::PoleAtOrigin);

    // Pole on the negative real axis
    let negative = TransferFunction::<f64, Discrete>::from_polynomials(Polynomial(vec![1.0]), Polynomial(vec![1.0, 0.5]));
    assert_eq!(to_continuous(&negative, TS).unwrap_err(), MatchedZError::NegativeRealPole { re: -0.5 });

    // Improper G(s)
    let improper = TransferFunction::<f64, Continuous>::from_polynomials(Polynomial(vec![1.0, 1.0, 1.0]), Polynomial(vec![1.0, 2.0]));
    assert_eq!(improper.discretize(MatchedZ(ZerosAtInfinity::MinusOne), TS).unwrap_err(), MatchedZError::Improper);
}

/// Perturbed zeros near z = -1 (as in identified models) are zeros at s = ∞ within the tolerance.
#[test]
fn perturbed_nyquist_zeros() {
    let g = tf!("1000 / (s^2 + 20s + 1000)");
    let gz = g.discretize(MatchedZ(ZerosAtInfinity::MinusOne), TS).unwrap();
    // (z + 1)^2 -> (z + 1)^2 + 1e-6: zeros at -1 ± 1e-3 j, mapped to |Im s| ≈ π / ts without the tolerance
    let mut numer = gz.numerator.clone();
    let last = numer.len() - 1;
    numer[last] += 1e-6 * numer[0];
    let perturbed = TransferFunction::<f64, Discrete>::from_polynomials(numer, gz.denominator.clone());

    let spurious = to_continuous(&perturbed, TS).unwrap();
    assert_eq!(normalized(&spurious).0.len(), 3, "a spurious zero pair near the Nyquist frequency: {spurious:?}");
    let back = to_continuous_with(&perturbed, TS, &ToContinuousOptions { nyquist_tolerance:1e-2}).unwrap();
    let (n0, d0) = normalized(&g);
    let (n1, d1) = normalized(&back);
    assert_coeffs_close(&d1, &d0, 1e-9, "denominator");
    assert_coeffs_close(&n1, &n0, 1e-5, "numerator");
}

/// A model of mismatched order (as identified with too few numerator coefficients) has zeros at
/// z = 0 and far out on the negative real axis; they are dropped, keeping the DC gain.
#[test]
fn unmappable_zeros_are_dropped() {
    let g = tf!("1000 / (s^2 + 20s + 1000)");
    let gz = g.discretize(MatchedZ(ZerosAtInfinity::MinusOne), TS).unwrap();
    // Replace the numerator by b0 z^2 + b1 z = b0 z (z + 102) with the same DC gain.
    let dc = gz.numerator.iter().sum::<f64>();
    let numer = Polynomial(vec![dc / 103.0, dc * 102.0 / 103.0, 0.0]);
    let mismatched = TransferFunction::<f64, Discrete>::from_polynomials(numer, gz.denominator.clone());

    let back = to_continuous(&mismatched, TS).unwrap();
    assert_same_tf(&back, &g, 1e-9, "zeros dropped");

    // Only zeros: a pole at z = 0 or on the negative real axis is still an error.
    let pole_negative = TransferFunction::<f64, Discrete>::from_polynomials(Polynomial(vec![1.0]), Polynomial(vec![1.0, 0.5]));
    assert!(matches!(to_continuous(&pole_negative, TS), Err(MatchedZError::NegativeRealPole { .. })));
}

/// Poles at z = 0 (input delay z^-d) become a dead time e^(-d ts s).
#[test]
fn dead_time() {
    let g = tf!("(1 - 0.01s) / ((s + 20)^2)");
    let delayed = dsmc::TransferFunctionWithDelay { tf: g.clone(), delay: 3.0 * TS };
    let gz = delayed.discretize(MatchedZ(ZerosAtInfinity::KeepOneDelay), TS).unwrap();
    // z^-3: three more trailing zeros in the denominator than without the delay
    let plain = g.discretize(MatchedZ(ZerosAtInfinity::KeepOneDelay), TS).unwrap();
    assert_eq!(gz.denominator.len(), plain.denominator.len() + 3);

    assert_eq!(to_continuous(&gz, TS).unwrap_err(), MatchedZError::PoleAtOrigin);
    let back = to_continuous_with_delay(&gz, TS, &ToContinuousOptions::default()).unwrap();
    assert!((back.delay - 3.0 * TS).abs() < 1e-15);
    assert_same_tf(&back.tf, &g, 1e-7, "dead time");

    // Frequency response includes the delay: phase -ω delay on top of tf(jω).
    let w = 50.0;
    let rational = g.frequency_transfer_function().response(w);
    assert!((back.frequency_response(w) - rational * Complex::new(0.0, -w * 3.0 * TS).exp()).norm() < 1e-9 * rational.norm());

    // A fractional delay has no z^-d form.
    let fractional = dsmc::TransferFunctionWithDelay { tf: g, delay: 2.5 * TS };
    assert!(matches!(fractional.discretize(MatchedZ(ZerosAtInfinity::MinusOne), TS), Err(MatchedZError::FractionalDelay { .. })));
}

/// Zeros at z = 0 are a time advance (negative delay) instead of being dropped.
#[test]
fn time_advance() {
    let g = tf!("1000 / (s^2 + 20s + 1000)");
    let gz = g.discretize(MatchedZ(ZerosAtInfinity::MinusOne), TS).unwrap();
    // b0 z (z + 102) with the same DC gain (as identified with a mismatched order)
    let dc = gz.numerator.iter().sum::<f64>();
    let numer = Polynomial(vec![dc / 103.0, dc * 102.0 / 103.0, 0.0]);
    let mismatched = TransferFunction::<f64, Discrete>::from_polynomials(numer, gz.denominator.clone());

    let back = to_continuous_with_delay(&mismatched, TS, &ToContinuousOptions::default()).unwrap();
    assert!((back.delay + TS).abs() < 1e-15, "delay {}", back.delay);
    assert_same_tf(&back.tf, &g, 1e-9, "time advance");
    assert_eq!(format!("{:.3}", back), "exp(0.001 s) * (1000.000 / (s^2 + 20.000 * s + 1000.000))");
}
