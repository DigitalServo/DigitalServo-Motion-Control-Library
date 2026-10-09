//! Identification from frequency responses: Levy and vector fitting.

use dsmc::TransferFunction;

use super::*;

#[test]
fn test_levy() {

    use nalgebra::Complex;
    use dsmc::FrequencyResponse;
    use dsmc::system_identification::frequency_response::levy;

    fn system(omega: f64) -> Complex<f64> {
        let g = Complex::new(100.0, 0.0);
        let s = Complex::new(0.0, omega);
        let numer = -3.0 * s.powi(2) + 10.0 * s.powi(1) + g;
        let denom = 2.0 * s.powi(3) + 4.0 * s.powi(2) + (2.0 * g) * s.powi(1) + g * g;
        numer / denom
    }

    let size = 100;
    let mut samples: Vec<FrequencyResponse<f64>> = Vec::with_capacity(size);

    let numer_order = 2;
    let denom_order = 3;

    for i in 0..size {
        let omega = 2.0 * std::f64::consts::PI * (i as f64);
        let value = system(omega);
        samples.push(FrequencyResponse { omega, value });
    }

    let ret = levy::sanathanan_koerner_identification(&samples, denom_order, numer_order, 5).unwrap();

    // True system: (-3s^2 + 10s + 100) / (2s^3 + 4s^2 + 200s + 10000)
    let expected = TransferFunction::continuous(&[-3.0, 10.0, 100.0], &[2.0, 4.0, 200.0, 10000.0]);
    let err = tf_distance(&ret, &expected);
    assert!(err < TOL_LEVY, "Levy: {ret:?}");

}

#[test]
fn test_vector_fitting() {
    use std::f64::consts::PI;
    use dsmc::{FrequencyResponse, system_identification::frequency_response::vector_fitting::{VectorFittingOptions, VectorFittingResult, identify}};
    use num_complex::Complex64;

    // poles and residues of the true function
    let true_poles = [
        Complex64::new(-0.0, 1.0),
        Complex64::new(-0.0, -1.0),
    ];
    let true_residues = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];

    // samples
    // DONOT INCLUDE DC Component freq = 0
    let freqs: Vec<f64> = (1..=1000)
        .map(|k| 0.1 * k as f64)
        .collect();

    let samples: Vec<FrequencyResponse<f64>> = freqs
        .iter()
        .map(|&f| {
            let s = Complex64::new(0.0, 2.0 * PI * f);
            let mut h = Complex64::new(0.0, 0.0);
            for (&a, &c) in true_poles.iter().zip(true_residues.iter()) {
                h += c / (s - a);
            }
            FrequencyResponse { omega: s.im, value: h }
        })
        .collect();

    let mut rms = f64::INFINITY;
    let mut ret: Option<(usize, VectorFittingResult<f64>)> = None;
    for order in 1..=5 {
        let opts = VectorFittingOptions { fit_d: true, fit_e: true, ..VectorFittingOptions::default() };
        let result = identify(&samples, order, &opts).unwrap();

        if rms > *result.rms_errors.last().unwrap() {
            rms = *result.rms_errors.last().unwrap();
            ret = Some((order, result))
        }
    }

    let (order, result) = ret.unwrap();

    assert!(
        rms < 1e-4,
        "RMS error too large: {:.2e}",
        rms
    );

    // The true system has two poles ±j with residues 1.
    assert_eq!(order, 2);
    for (&p, &r) in true_poles.iter().zip(true_residues.iter()) {
        let k = result.poles.iter().position(|&q| (q - p).norm() < 1e-8).expect("pole not found");
        assert!((result.residues[k] - r).norm() < 1e-8, "residue {} != {}", result.residues[k], r);
    }

    // True system: 1/(s - j) + 1/(s + j) = 2s / (s^2 + 1)
    let tf: TransferFunction<f64> = result.into();
    let expected = TransferFunction::continuous(&[2.0, 0.0], &[1.0, 0.0, 1.0]);
    let err = tf_distance(&tf, &expected);
    assert!(err < TOL_VF, "Vector fitting: {tf:?}");

}

/// Conversion of a vector fitting result into a transfer function: the leading numerator
/// coefficients that cancel are dropped relative to the size of the other terms, whatever the gain
/// (units of G) and the frequency scale.
#[test]
fn test_vector_fitting_into_transfer_function_scale() {
    use dsmc::system_identification::frequency_response::vector_fitting::VectorFittingResult;
    use num_complex::Complex64;

    // G = k wn^2 / (s^2 + 2 ζ wn s + wn^2) = r / (s - p) + r* / (s - p*), r = k wn^2 / (p - p*)
    for (k, wn) in [(1.0, 1.0), (1e-10, 100.0), (1e-10, 1e4), (1e3, 1e-3), (1.0, 1e5)] {
        let zeta = 0.1;
        let p = Complex64::new(-zeta * wn, wn * (1.0 - zeta * zeta).sqrt());
        let r = k * wn * wn / (p - p.conj());
        let result = VectorFittingResult { poles: vec![p, p.conj()], residues: vec![r, r.conj()], d: 0.0, e: 0.0, rms_errors: vec![] };
        let tf: TransferFunction<f64> = result.into();
        let expected = TransferFunction::continuous(&[k * wn * wn], &[1.0, 2.0 * zeta * wn, wn * wn]);
        assert_eq!(tf.numerator.len(), 1, "k = {k}, wn = {wn}: {tf:?}");
        assert!((tf.numerator[0] / tf.denominator[0] - k * wn * wn).abs() < 1e-9 * k * wn * wn, "k = {k}, wn = {wn}: {tf:?}");
        assert!(tf_distance(&tf, &expected) < 1e-9 * wn * wn, "k = {k}, wn = {wn}: {tf:?}");
    }
}
