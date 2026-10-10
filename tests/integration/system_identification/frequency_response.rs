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

/// Samples of the noisy responses of the vector fitting tests: `G(jω)` at `omegas` with a
/// multiplicative complex error `1 + ε`, `|ε|` uniform in `[low, high]` and of uniform phase
/// (xorshift64).
fn noisy_samples(g: &TransferFunction<f64>, omegas: &[f64], (low, high): (f64, f64)) -> Vec<dsmc::FrequencyResponse<f64>> {
    use num_complex::Complex64;
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let response = g.frequency_transfer_function();
    omegas
        .iter()
        .map(|&omega| {
            let magnitude = low + (high - low) * uniform();
            let error = Complex64::from_polar(magnitude, 2.0 * std::f64::consts::PI * uniform());
            dsmc::FrequencyResponse { omega, value: response.response(omega) * (1.0 + error) }
        })
        .collect()
}

/// Checks of a vector fitting result on noisy data: the residues of the pairs are exactly
/// conjugate and those of the real poles real, every true pole is matched within `tolerance`
/// (relative), and the iterations converge with a settled RMS error (no oscillation).
fn check_real_fit(result: &dsmc::system_identification::frequency_response::vector_fitting::VectorFittingResult<f64>, true_poles: &[num_complex::Complex64], tolerance: f64) {
    println!("poles {:?}\nresidues {:?}\nrms {:?}", result.poles, result.residues, result.rms_errors);
    let mut k = 0;
    while k < result.poles.len() {
        let p = result.poles[k];
        if p.im == 0.0 {
            assert_eq!(result.residues[k].im, 0.0, "residue of the real pole {p}");
            k += 1;
        } else {
            assert!(p.im > 0.0 && result.poles[k + 1] == p.conj(), "pair {p}, {}", result.poles[k + 1]);
            assert_eq!(result.residues[k + 1], result.residues[k].conj(), "residues of the pair {p}");
            k += 2;
        }
    }
    for p in true_poles {
        let error = result.poles.iter().map(|q| (q - p).norm() / p.norm()).fold(f64::INFINITY, f64::min);
        assert!(error < tolerance, "pole {p}: relative error {error:.3e}");
    }
    // Converged before `max_iter` (30), the RMS error settled
    let rms = &result.rms_errors;
    assert!(rms.len() < 30 && (rms[rms.len() - 1] / rms[rms.len() - 2] - 1.0).abs() < 1e-4, "rms errors do not settle: {rms:?}");
}

/// Noisy samples (multiplicative errors of 1 .. 5 %) of `1000 / (s^2 + 20 s + 1000)`: the fit of
/// two poles is a conjugate pair near `-10 ± 30j` with conjugate residues, the iterations
/// converge, and the transfer function matches the plant.
#[test]
fn test_vector_fitting_noisy_pair() {
    use dsmc::system_identification::frequency_response::vector_fitting::{identify, VectorFittingOptions};
    use num_complex::Complex64;

    let g = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let omegas: Vec<f64> = (1..=150).map(|k| 2.0 * std::f64::consts::PI * 0.1 * k as f64).collect();
    let samples = noisy_samples(&g, &omegas, (0.01, 0.05));
    let opts = VectorFittingOptions { max_iter: 30, ..VectorFittingOptions::default() };
    let result = identify(&samples, 2, &opts).unwrap();
    check_real_fit(&result, &[Complex64::new(-10.0, 30.0), Complex64::new(-10.0, -30.0)], 0.03);
    let tf: TransferFunction<f64> = result.into();
    println!("{tf}");
    let k = tf.denominator[0];
    let relative = |a: f64, b: f64| (a / b - 1.0).abs();
    assert!(relative(tf.denominator[1] / k, 20.0) < 0.1 && relative(tf.denominator[2] / k, 1000.0) < 0.03, "{tf:?}");
    assert!(relative(tf.numerator[tf.numerator.len() - 1] / k, 1000.0) < 0.03, "{tf:?}");
}

/// One real pole, `5 / (s + 5)` with errors of 1 .. 5 %: the residue is real and the pole near -5.
#[test]
fn test_vector_fitting_noisy_real_pole() {
    use dsmc::system_identification::frequency_response::vector_fitting::{identify, VectorFittingOptions};
    use num_complex::Complex64;

    let g = TransferFunction::continuous(&[5.0], &[1.0, 5.0]);
    let omegas: Vec<f64> = (1..=150).map(|k| 0.3 * k as f64).collect();
    let samples = noisy_samples(&g, &omegas, (0.01, 0.05));
    let opts = VectorFittingOptions { max_iter: 30, ..VectorFittingOptions::default() };
    let result = identify(&samples, 1, &opts).unwrap();
    check_real_fit(&result, &[Complex64::new(-5.0, 0.0)], 0.03);
    assert!((result.residues[0].re / 5.0 - 1.0).abs() < 0.03, "residue {}", result.residues[0]);
}

/// Band-limited samples of a low-order model with a high resonance left out:
/// `1000 / (s^2 + 20 s + 1000)` times a resonance at 1000 rad/s, sampled up to 100 rad/s with
/// errors of 1 .. 5 %. Two poles fit the low mode (within the quasi-static tail of the resonance,
/// `(ω / 1000)^2` <= 1 %).
#[test]
fn test_vector_fitting_noisy_band_limited() {
    use dsmc::system_identification::frequency_response::vector_fitting::{identify, VectorFittingOptions};
    use num_complex::Complex64;

    let wr: f64 = 1000.0;
    let g = &TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]) * &TransferFunction::continuous(&[wr * wr], &[1.0, 0.04 * wr, wr * wr]);
    let omegas: Vec<f64> = (1..=150).map(|k| 100.0 * k as f64 / 150.0).collect();
    let samples = noisy_samples(&g, &omegas, (0.01, 0.05));
    let opts = VectorFittingOptions { max_iter: 30, ..VectorFittingOptions::default() };
    let result = identify(&samples, 2, &opts).unwrap();
    check_real_fit(&result, &[Complex64::new(-10.0, 30.0), Complex64::new(-10.0, -30.0)], 0.03);
}

/// The case of the report: `1000 / (s^2 + 20 s + 1000)` measured by Welch's method (multisine of
/// period 1024 samples with a flat output spectrum, `ts` = 1 ms, 4 segments, white noise of 3 %
/// of the output RMS), the 150 excited bins. Against `G(jω)` they are off by 23 % (median), up to
/// 45 %: mostly the delay of the held input, `e^(-jω ts / 2)` (0.46 rad at 146 Hz), which no
/// rational model without delay fits, and the leakage of the window (up to 26 % without the
/// delay, median 2 %). Independent complex residues oscillated on such data (RMS 0.33 .. 1.02) to
/// poles far off with residues that were not conjugate; the real formulation converges to a
/// conjugate pair within 5 % of -10 ± 30j.
#[test]
fn test_vector_fitting_welch_measurement() {
    use dsmc::fft::welch;
    use dsmc::signal::excitation::flat_output_multisine;
    use dsmc::system_identification::frequency_response::vector_fitting::{identify, VectorFittingOptions};
    use dsmc::TransferFunctionWithDelay;
    use num_complex::Complex64;

    let (ts, period) = (1e-3, 1024);
    let g = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let harmonics: Vec<usize> = (1..=150).collect();
    let u: Vec<f64> = flat_output_multisine(&g, 4.0 * period as f64 * ts, ts, 1.0 / (period as f64 * ts), &harmonics, 7).unwrap();
    let y0 = TransferFunctionWithDelay::from(&g).simulate(ts, &u).unwrap();
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut gaussian = move || {
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos()
    };
    let scale = 0.03 * (y0.iter().map(|v| v * v).sum::<f64>() / y0.len() as f64).sqrt();
    let y: Vec<f64> = y0.iter().map(|v| v + scale * gaussian()).collect();
    let (response, _) = welch(&u, &y, ts, 4);
    let samples: Vec<_> = response.into_iter().skip(1).take(150).collect();
    let response_true = g.frequency_transfer_function();
    // Errors against G(jω), and against the response of the held input, G(jω) e^(-jω ts / 2) to
    // second order (the noise alone)
    let summary = |errors: Vec<f64>| {
        let mut sorted = errors;
        sorted.sort_by(f64::total_cmp);
        (sorted[sorted.len() / 2], sorted[sorted.len() - 1])
    };
    let against = |delay: f64| summary(samples.iter().map(|s| (s.value / (response_true.response(s.omega) * Complex64::from_polar(1.0, -s.omega * delay)) - 1.0).norm()).collect());
    println!("errors (median, max): against G {:.3?}, against G e^(-jω ts / 2) {:.3?}", against(0.0), against(ts / 2.0));

    let opts = VectorFittingOptions { max_iter: 30, ..VectorFittingOptions::default() };
    let result = identify(&samples, 2, &opts).unwrap();
    check_real_fit(&result, &[Complex64::new(-10.0, 30.0), Complex64::new(-10.0, -30.0)], 0.05);
}

/// `identify_weighted` by `1 / |G|` (relative error) on a response over three decades of gain:
/// the high-frequency bins, which hardly count in the absolute error, are fitted as well as the
/// low ones. Invalid weights are an error.
#[test]
fn test_vector_fitting_weighted() {
    use dsmc::system_identification::frequency_response::vector_fitting::{identify, identify_weighted, VectorFittingError, VectorFittingOptions};
    use num_complex::Complex64;

    // Two real poles at 1 and 100 rad/s: 100 / ((s + 1) (s + 100)), 0.1 .. 1000 rad/s
    let g = TransferFunction::continuous(&[100.0], &[1.0, 101.0, 100.0]);
    let omegas: Vec<f64> = (0..150).map(|k| 0.1 * 10f64.powf(4.0 * k as f64 / 149.0)).collect();
    let samples = noisy_samples(&g, &omegas, (0.01, 0.05));
    let weights: Vec<f64> = samples.iter().map(|s| 1.0 / s.value.norm()).collect();
    let opts = VectorFittingOptions { max_iter: 30, ..VectorFittingOptions::default() };
    let relative = identify_weighted(&samples, &weights, 2, &opts).unwrap();
    let absolute = identify(&samples, 2, &opts).unwrap();
    check_real_fit(&relative, &[Complex64::new(-1.0, 0.0), Complex64::new(-100.0, 0.0)], 0.05);
    let fast = |r: &dsmc::system_identification::frequency_response::vector_fitting::VectorFittingResult<f64>| {
        r.poles.iter().map(|p| p.norm()).fold(0.0f64, f64::max)
    };
    println!("fastest pole: weighted {:.2}, unweighted {:.2}", fast(&relative), fast(&absolute));
    assert!((fast(&relative) / 100.0 - 1.0).abs() < (fast(&absolute) / 100.0 - 1.0).abs());

    assert!(matches!(identify_weighted(&samples, &weights[1..], 2, &opts), Err(VectorFittingError::InvalidWeights { len: 149, samples: 150 })));
    let mut negative = weights.clone();
    negative[3] = -1.0;
    assert!(matches!(identify_weighted(&samples, &negative, 2, &opts), Err(VectorFittingError::InvalidWeights { .. })));
}
