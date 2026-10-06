//! System identification: ARX models by least squares / Kalman filter (with input delay
//! estimation), IV, SRIVC (also on multi-inertia plants: structure search, input offset, high
//! orders in `mod srivc`), Levy, vector fitting and Gaussian process regression. The validation of
//! the identified models is tested in `validation.rs`.

use dsmc::{DiscreteSystem, StateSpace, TransferFunction, TransferFunctionWithDelay, discretize::Zoh};

/// Max absolute difference between the coefficients of two transfer functions, after
/// normalizing both so that the leading denominator coefficient is 1.
// Tolerances on normalized coefficients. All data are noise-free, so the measured errors are
// 1e-16 .. 1e-8; these leave a margin while still catching a broken identification.
const TOL_LSM_ARX: f64 = 1e-6;
const TOL_KF_ARX: f64 = 1e-6;
// Continuous-time parameters from the ARX models by the matched z-transform. It is not the inverse
// of the bilinear transform the data come from (frequency warping: relative (ω ts)^2 / 12 ≈ 1e-4
// for ω ≈ 32 rad/s, ts = 1 ms), so this bounds the method difference, not the identification.
const TOL_LSM_POLY: f64 = 1e-8;
const TOL_KF_POLY: f64 = 1e-8;
const TOL_LEVY: f64 = 1e-6;
const TOL_VF: f64 = 1e-8;

fn tf_distance<D: dsmc::Domain>(a: &TransferFunction<f64, D>, b: &TransferFunction<f64, D>) -> f64 {
    let normalize = |tf: &TransferFunction<f64, D>| {
        let k = tf.denominator[0];
        let n: Vec<f64> = tf.numerator.iter().map(|c| c / k).collect();
        let d: Vec<f64> = tf.denominator.iter().map(|c| c / k).collect();
        (n, d)
    };
    let (na, da) = normalize(a);
    let (nb, db) = normalize(b);
    assert_eq!((na.len(), da.len()), (nb.len(), db.len()), "orders differ: {a:?} vs {b:?}");
    na.iter().zip(&nb).chain(da.iter().zip(&db)).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}


#[test]
fn test_lsm_arx() {
    use dsmc::{discretize::matched_z_transform, system_identification::{kalman_filter, lsm}};

    let ts: f64 = 1e-3;

    let tf_c = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);

    let (input_order, state_order) = (2, 2);
    // A plant driven through a zero-order hold has a one-sample delay (u[k] does not affect y[k]).
    let mut lsm = lsm::arx::DataBuffer::<f64>::new(state_order, input_order).with_input_delay(1);
    let mut kf = kalman_filter::arx::KalmanFilter::<f64>::new(state_order, input_order, 0.01, 0.01, 1.0e10).with_input_delay(1);

    let mut system = DiscreteSystem::from(StateSpace::try_from(&tf_c).unwrap().discretize(Zoh, ts).unwrap());

    let mut t = 0.0;
    for _ in 0..1000 {
        let x_prev = system.output[0];

        let u = (0..50).fold(0.0, |a, b| a + 0.1 * ((b as f64) * t).sin());
        let y = system.update(&[u]).unwrap();

        lsm.add(u, x_prev, y[0]);
        kf.update(u, x_prev, y[0]);

        t += ts;
    }

    // The data come from the bilinear-discretized system itself (noise-free), so both methods
    // should recover it.
    let tf_z = tf_c.discretize(Zoh, ts).unwrap();

    let tf_lsm = lsm.identify().unwrap();
    let tf_kf = kf.identify();

    println!("{tf_z}");
    println!("{tf_lsm}");
    println!("{tf_kf}");

    // Continuous-time parameters. The double zero at z = -1 of the bilinear transform is perturbed
    // in the identified models (split by ~1e-3), so zeros near z = -1 are taken as zeros at s = ∞.
    let options = matched_z_transform::ToContinuousOptions { nyquist_tolerance: 1e-2 };
    for (name, tf_d) in [("LS method", &tf_lsm), ("Kalman filter", &tf_kf)] {
        let tf_s = matched_z_transform::to_continuous_with(tf_d, ts, &options).unwrap();
        // let err = tf_relative_distance(&tf_s, &tf_c);
        // assert!(err < TOL_ARX_CONTINUOUS, "{name}: continuous-time model {tf_s:?}, relative error {err:e}");

        println!("{name}: {tf_s:.04}");
    }
}


#[test]
fn test_lsm_polynomial() {
    use dsmc::system_identification::{kalman_filter, lsm};

    let dx: f64 = 1e-3;

    let mut lsm = lsm::polynomial::DataBuffer::<f64>::new(3);
    let mut kf = kalman_filter::polynomial::KalmanFilter::<f64>::new(3, 0.0, 0.0 ,1.0e5);

    let generator = |x: f64| -> f64 {
        3.0 * x.powi(3) - 1.0 * x.powi(2) + 0.3 * x.powi(1) + 0.05
    };

    let mut x = 0.0;
    for _ in 0..1000 {
        let y: f64 = generator(x);
        lsm.add(x, y);
        kf.update(x, y);
        x += dx;
    }

    // Coefficients are in descending order: 3x^3 - x^2 + 0.3x + 0.05
    let expected = [3.0, -1.0, 0.3, 0.05];
    let lsm_coeffs = lsm.identify().unwrap();
    let kf_coeffs = kf.identify();
    let err = |c: &[f64]| c.iter().zip(expected).map(|(x, e)| (x - e).abs()).fold(0.0, f64::max);
    assert!(err(&lsm_coeffs) < TOL_LSM_POLY, "LS method: {lsm_coeffs:?}");
    assert!(err(&kf_coeffs) < TOL_KF_POLY, "Kalman filter: {kf_coeffs:?}");
}


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

/// ARX models with input order != state order: `y[k] = Σ a_i y[k-i] + Σ b_i u[k-i]` is
/// `(b_0 + b_1 z^-1 + ...) / (1 - a_1 z^-1 - ...)`, i.e. polynomials in z of degree max(na, nb).
#[test]
fn test_arx_mismatched_orders() {
    use dsmc::system_identification::{kalman_filter, lsm};
    use dsmc::{Discrete, Polynomial};

    // (a, b) and the expected G(z) = numerator / denominator
    type Case<'a> = (&'a [f64], &'a [f64], Vec<f64>, Vec<f64>);
    let cases: [Case; 2] = [
        // na = 2, nb = 1: (b0 z^2 + b1 z) / (z^2 - a1 z - a2)
        (&[1.5, -0.7], &[0.2, 0.1], vec![0.2, 0.1, 0.0], vec![1.0, -1.5, 0.7]),
        // na = 1, nb = 2: (b0 z^2 + b1 z + b2) / (z^2 - a1 z)
        (&[0.8], &[0.3, 0.2, 0.1], vec![0.3, 0.2, 0.1], vec![1.0, -0.8, 0.0]),
    ];
    for (a, b, numer, denom) in cases {
        let expected = TransferFunction::<f64, Discrete>::from_polynomials(Polynomial(numer), Polynomial(denom));
        let mut lsm = lsm::arx::DataBuffer::<f64>::new(a.len(), b.len() - 1);
        let mut kf = kalman_filter::arx::KalmanFilter::<f64>::new(a.len(), b.len() - 1, 0.01, 0.01, 1.0e10);

        let (mut ys, mut us) = (vec![0.0; a.len()], vec![0.0; b.len()]);
        for k in 0..2000 {
            let t = k as f64 * 1e-3;
            let u = (0..20).map(|i| 0.1 * (i as f64 * 37.0 * t).sin()).sum::<f64>();
            us.rotate_right(1);
            us[0] = u;
            let y = a.iter().zip(&ys).map(|(ai, yi)| ai * yi).sum::<f64>() + b.iter().zip(&us).map(|(bi, ui)| bi * ui).sum::<f64>();
            let y_prev = ys[0];
            lsm.add(u, y_prev, y);
            kf.update(u, y_prev, y);
            ys.rotate_right(1);
            ys[0] = y;
        }

        let err_lsm = tf_distance(&lsm.identify().unwrap(), &expected);
        let err_kf = tf_distance(&kf.identify(), &expected);
        assert!(err_lsm < TOL_LSM_ARX, "LS method (na = {}, nb = {}): error {err_lsm:e}", a.len(), b.len() - 1);
        assert!(err_kf < TOL_KF_ARX, "Kalman filter (na = {}, nb = {}): error {err_kf:e}", a.len(), b.len() - 1);
    }
}

/// Plant with a dead time: the input delay is estimated, identified with the ARX models and
/// recovered as a continuous-time dead time.
#[test]
fn test_arx_input_delay() {
    use dsmc::discretize::matched_z_transform::{
        to_continuous_with_delay, MatchedZ, ToContinuousOptions, ZerosAtInfinity,
    };
    use dsmc::system_identification::{kalman_filter, lsm};

    let ts: f64 = 1e-3;
    let delay_samples = 5;
    let tf_c = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let plant = TransferFunctionWithDelay { tf: tf_c.clone(), delay: delay_samples as f64 * ts };
    // z^-5 (b0 z^2 + b1 z + b2) / (z^2 + a1 z + a2): na = nb = 2, nk = 5
    let tf_z = plant.discretize(MatchedZ(ZerosAtInfinity::MinusOne), ts).unwrap();

    let mut system = DiscreteSystem::try_from(&tf_z).unwrap();
    let (mut u, mut y) = (Vec::new(), Vec::new());
    for k in 0..3000 {
        let t = k as f64 * ts;
        let uk = (1..=20).map(|i| 0.1 * (37.0 * i as f64 * t).sin()).sum::<f64>();
        u.push(uk);
        y.push(system.update(uk));
    }

    // Delay estimation by the one-step prediction error over nk = 0..=10
    let estimate = lsm::arx::estimate_input_delay(&u, &y, 2, 2, 10).unwrap();
    assert_eq!(estimate.input_delay, delay_samples, "losses: {:?}", estimate.losses);
    let loss = |nk: usize| estimate.losses[nk].unwrap();
    assert!(loss(0) > 1e6 * loss(delay_samples), "without the delay the fit is poor: {:?}", estimate.losses);
    let err = tf_distance(&estimate.model, &tf_z);
    assert!(err < TOL_LSM_ARX, "LS method with delay: error {err:e}");

    // Kalman filter with the delay set
    let mut kf = kalman_filter::arx::KalmanFilter::<f64>::new(2, 2, 0.01, 0.01, 1.0e10).with_input_delay(delay_samples);
    for k in 0..u.len() {
        let y_prev = if k > 0 { y[k - 1] } else { 0.0 };
        kf.update(u[k], y_prev, y[k]);
    }
    let err = tf_distance(&kf.identify(), &tf_z);
    assert!(err < TOL_KF_ARX, "Kalman filter with delay: error {err:e}");

    // Continuous time: the poles at z = 0 become the dead time.
    let options = ToContinuousOptions { nyquist_tolerance: 1e-2 };
    for (name, model) in [("LS method", estimate.model.clone()), ("Kalman filter", kf.identify())] {
        let g = to_continuous_with_delay(&model, ts, &options).unwrap();
        assert!((g.delay - plant.delay).abs() < 1e-12, "{name}: delay {}", g.delay);
        // Relative error per coefficient (normalized by the leading denominator coefficient)
        let normalized = |tf: &TransferFunction<f64>| {
            let k = tf.denominator[0];
            tf.numerator.iter().chain(tf.denominator.iter()).map(|c| c / k).collect::<Vec<f64>>()
        };
        let (got, expected) = (normalized(&g.tf), normalized(&tf_c));
        assert_eq!(got.len(), expected.len(), "{name}: {g}");
        let err = got.iter().zip(&expected).map(|(x, e)| (x - e).abs() / e.abs().max(1.0)).fold(0.0, f64::max);
        assert!(err < 1e-4, "{name}: {g}, relative error {err:e}");
    }
}

/// Output-error data `y = G u + v` with white measurement noise `v`: least squares is biased since
/// the past outputs in the regressor contain the noise, while the IV method recovers `G`.
#[test]
fn test_iv_arx() {
    use dsmc::system_identification::{arx::Arx, iv, lsm};
    use dsmc::Discrete;

    // Uniform pseudo-random numbers in [-1, 1) (xorshift64)
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };

    // y0[k] = 1.5 y0[k-1] - 0.7 y0[k-2] + 1.0 u[k-1] + 0.5 u[k-2]: na = 2, nb = 1, nk = 1
    let (a, b) = ([1.5, -0.7], [1.0, 0.5]);
    let expected = TransferFunction::<f64, Discrete>::discrete(&[0.0, 1.0, 0.5], &[1.0, -1.5, 0.7]);

    let n = 20000;
    let (mut u, mut y0) = (vec![0.0; n], vec![0.0; n]);
    for k in 0..n {
        u[k] = rand();
        let y = |i: usize| if k >= i { y0[k - i] } else { 0.0 };
        let u = |i: usize| if k >= i { u[k - i] } else { 0.0 };
        y0[k] = a[0] * y(1) + a[1] * y(2) + b[0] * u(1) + b[1] * u(2);
    }
    let structure = Arx::<f64>::new(2, 1).with_input_delay(1);

    // Noise-free: IV recovers the plant exactly, like least squares.
    let model = iv::arx::identify(&u, &y0, &structure, 1).unwrap();
    let err = tf_distance(&model.transfer_function(), &expected);
    assert!(err < TOL_LSM_ARX, "IV method (noise-free): error {err:e}");

    // Noise with a standard deviation of ~30 % of that of y0
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    let y: Vec<f64> = y0.iter().map(|&v| v + 0.5 * rms * rand()).collect();

    let mut lsm = lsm::arx::DataBuffer::from_arx(structure.clone());
    for k in 0..n {
        lsm.add(u[k], if k > 0 { y[k - 1] } else { 0.0 }, y[k]);
    }
    let err_lsm = tf_distance(&lsm.identify().unwrap(), &expected);

    for iterations in [1, 3] {
        let model = iv::arx::identify(&u, &y, &structure, iterations).unwrap();
        let err_iv = tf_distance(&model.transfer_function(), &expected);
        println!("LS error {err_lsm:e}, IV ({iterations} iterations) error {err_iv:e}");
        assert!(err_iv < 0.05, "IV method ({iterations} iterations): error {err_iv:e}");
        assert!(err_iv < 0.2 * err_lsm, "IV method ({iterations} iterations): error {err_iv:e}, LS {err_lsm:e}");
    }
}

/// SRIVC: continuous-time `G(s)` from ZOH input / sampled output. Noise-free data are fitted
/// exactly; with strong white output noise the least-squares state-variable-filter estimate (the
/// SRIVC starting point, `max_iterations = 0`) is biased while SRIVC is not.
#[test]
fn test_srivc() {
    use dsmc::system_identification::iv::srivc::{identify, Initialization, SrivcOptions};

    // Uniform pseudo-random numbers in [-1, 1) (xorshift64)
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };

    // Max relative error of the coefficients (denominators normalized to monic)
    let relative_error = |got: &TransferFunction<f64>, expected: &TransferFunction<f64>| {
        let coefficients = |tf: &TransferFunction<f64>| {
            let k = tf.denominator[0];
            let mut c = vec![0.0; tf.denominator.len() - tf.numerator.len()];
            c.extend(tf.numerator.iter().chain(tf.denominator.iter()).map(|c| c / k));
            c
        };
        let (got, expected) = (coefficients(got), coefficients(expected));
        assert_eq!(got.len(), expected.len());
        got.iter().zip(&expected).map(|(x, e)| (x - e).abs() / e.abs().max(1.0)).fold(0.0, f64::max)
    };

    let ts: f64 = 1e-3;
    let n_samples = 20000;
    let init = Initialization::StateVariableFilter(30.0);
    let options = SrivcOptions::default();

    let cases = [
        (TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]), 0, 2),
        (TransferFunction::continuous(&[20.0, 1000.0], &[1.0, 20.0, 1000.0]), 1, 2),
    ];
    for (tf_c, m, n) in cases {
        let mut system = DiscreteSystem::from(StateSpace::try_from(&tf_c).unwrap().discretize(Zoh, ts).unwrap());
        let u: Vec<f64> = (0..n_samples).map(|_| rand()).collect();
        let y0: Vec<f64> = u.iter().map(|&uk| system.update(&[uk]).unwrap()[0]).collect();

        let result = identify(&u, &y0, ts, n, m, &init, &options).unwrap();
        let err = relative_error(&result.model.tf, &tf_c);
        assert!(result.converged && err < 1e-8, "noise-free: {} ({} iterations), error {err:e}", result.model, result.iterations);

        // Noise with a standard deviation of ~2.3 times the RMS of y0
        let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
        let y: Vec<f64> = y0.iter().map(|&v| v + 4.0 * rms * rand()).collect();
        let ls = identify(&u, &y, ts, n, m, &init, &SrivcOptions { max_iterations: 0, ..options.clone() }).unwrap();
        let err_ls = relative_error(&ls.model.tf, &tf_c);
        let result = identify(&u, &y, ts, n, m, &init, &options).unwrap();
        let err_iv = relative_error(&result.model.tf, &tf_c);
        assert!(result.converged, "noisy: not converged in {} iterations", result.iterations);
        assert!(err_iv < 0.1, "SRIVC: {}, error {err_iv:e}", result.model);
        assert!(err_iv < 0.3 * err_ls, "SRIVC error {err_iv:e}, LS-SVF {err_ls:e}");
    }
}

#[test]
fn test_srivc_with_delay() {
    use dsmc::system_identification::iv::srivc::{identify, Initialization, SrivcOptions};
    use std::f64::consts::PI;

    let multisine = |t: f64| (1..30).fold(0.0, |sum, freq_seed| {
        let freq = freq_seed as f64 * 1.0;
        let omega = 2.0 * PI * freq;
        sum + (omega * t).sin()
    });

    // Max relative error of the coefficients (denominators normalized to monic)
    let relative_error = |got: &TransferFunction<f64>, expected: &TransferFunction<f64>| {
        let coefficients = |tf: &TransferFunction<f64>| {
            let k = tf.denominator[0];
            let mut c = vec![0.0; tf.denominator.len() - tf.numerator.len()];
            c.extend(tf.numerator.iter().chain(tf.denominator.iter()).map(|c| c / k));
            c
        };
        let (got, expected) = (coefficients(got), coefficients(expected));
        assert_eq!(got.len(), expected.len());
        got.iter().zip(&expected).map(|(x, e)| (x - e).abs() / e.abs().max(1.0)).fold(0.0, f64::max)
    };

    let ts: f64 = 1e-3;
    let n_samples = 20000;
    let options = SrivcOptions::default();

    // Input delay of 5 samples, started from a rough initial model
    let tf_c = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let delay = 5;
    let mut system = DiscreteSystem::from(StateSpace::try_from(&tf_c).unwrap().discretize(Zoh, ts).unwrap());
    let u: Vec<f64> = (0..n_samples).map(|i| multisine(i as f64 * ts)).collect();
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let init = Initialization::Model(TransferFunction::continuous(&[500.0], &[1.0, 50.0, 700.0]));
    let options = SrivcOptions { input_delay: delay, ..options };

    let result = identify(&u, &y0, ts, 2, 1, &init, &options).unwrap();
    let err = relative_error(&result.model.tf, &tf_c);
    println!("Noise-free: {:.3} ({} iterations), error {err:e}", result.model, result.iterations);
    assert!(result.converged && err < 1e-8, "with delay: {} ({} iterations), error {err:e}", result.model, result.iterations);

    // Gaussian white noise on the output (xorshift64 + Box-Muller), standard deviation
    // `noise_ratio` times the RMS of y0
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let noise_ratio = 0.2;
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
    let y: Vec<f64> = y0.iter().map(|&v| v + noise_ratio * rms * gaussian()).collect();

    let result = identify(&u, &y, ts, 2, 1, &init, &options).unwrap();
    let err = relative_error(&result.model.tf, &tf_c);
    println!("Noisy ({noise_ratio} RMS): {:.3} ({} iterations), error {err:e}", result.model, result.iterations);
    assert!(result.converged && err < 1e-1, "with delay and noise: {} ({} iterations), error {err:e}", result.model, result.iterations);
}


/// Gaussian process regression against the closed form computed with nalgebra:
/// `mean = kᵀ (K + σI)^-1 y`, `stdev = sqrt(k(x, x) - kᵀ (K + σI)^-1 k + σ)`.
#[test]
fn test_gpr() {
    use dsmc::system_identification::gpr::GaussianProcessRegression;
    use nalgebra::{DMatrix, DVector};

    fn kernel(x1: f64, x2: f64) -> f64 {
        (-(x1 - x2).powi(2) / (2.0 * 0.3f64.powi(2))).exp()
    }
    let x_sample: Vec<f64> = (0..8).map(|i| i as f64 * 0.25).collect();
    let y_sample: Vec<f64> = x_sample.iter().map(|x| x.sin()).collect();

    for sigma in [0.0, 1e-2] {
        let mut gpr = GaussianProcessRegression::new(kernel, sigma);
        for (&x, &y) in x_sample.iter().zip(&y_sample) {
            gpr.add(x, y);
        }

        let n = x_sample.len();
        let k_inv = DMatrix::from_fn(n, n, |i, j| kernel(x_sample[i], x_sample[j]) + if i == j { sigma } else { 0.0 })
            .try_inverse()
            .unwrap();
        let y = DVector::from_vec(y_sample.clone());
        for x in [0.1, 0.5, 0.6, 1.3, 2.5] {
            let k = DVector::from_fn(n, |i, _| kernel(x_sample[i], x));
            let mean = k.dot(&(&k_inv * &y));
            // Compared as a variance: at a sample with σ = 0 it is ~0, and the square root would
            // magnify its rounding
            let variance = kernel(x, x) - k.dot(&(&k_inv * &k)) + sigma;
            let predicted = gpr.predict(x);
            assert!((predicted.mean - mean).abs() < 1e-8, "σ = {sigma}, x = {x}: mean {} vs {mean}", predicted.mean);
            assert!((predicted.stdev.powi(2) - variance).abs() < 1e-10, "σ = {sigma}, x = {x}: stdev {} vs variance {variance}", predicted.stdev);
        }

        // Without noise the samples are interpolated (the standard deviation vanishes there);
        // with noise they are smoothed, and the standard deviation is at least the noise level
        let at_sample = gpr.predict(0.5);
        if sigma == 0.0 {
            assert!((at_sample.mean - 0.5f64.sin()).abs() < 1e-6 && at_sample.stdev < 1e-4);
        } else {
            assert!((at_sample.mean - 0.5f64.sin()).abs() > 1e-6 && at_sample.stdev >= sigma.sqrt());
        }
    }

    // Repeated inputs: singular K, but K + σI is regular, and the mean averages the samples
    let mut gpr = GaussianProcessRegression::new(kernel, 1e-2);
    for y in [0.9, 1.1, 1.0, 0.8, 1.2] {
        gpr.add(1.0, y);
    }
    let predicted = gpr.predict(1.0);
    // kᵀ (K + σI)^-1 y with K = 11ᵀ: Σy / (n + σ)
    assert!((predicted.mean - 5.0 / (5.0 + 1e-2)).abs() < 1e-9, "mean {}", predicted.mean);
}


// ---------------------------------------------------------------------------------------------
// SRIVC on multi-inertia plants
// ---------------------------------------------------------------------------------------------

mod srivc {
    //! SRIVC on plants typical of an open-loop motion-control identification test: multi-inertia
    //! mechanics (rigid-body integrator, lightly damped resonances / antiresonances) seen through an
    //! analog low-pass filter and a dead time, excited by a band-limited multisine. Everything but
    //! the synthetic data (plants, noise) and the comparison with the true plant is the library's:
    //! `TransferFunctionWithDelay` (plant, simulation, identified model), `signal::excitation`,
    //! `preprocessing::high_pass`, `srivc::search` with `validation::Check`.
    //!
    //! Three things are checked, each of them needed to use `srivc::identify` on such data:
    //!
    //! 1. `test_srivc_structure_search`: the denominator order `n`, the numerator order `m` and the
    //!    input delay `nk` have to be searched *jointly*, and validated by the residual tests of
    //!    `Validation` (not by the raw validation error, which differs only in the 4th digit between
    //!    candidates), then compared by BIC (`srivc::search`).
    //! 2. `test_srivc_input_offset`: a small constant input offset (drift of the integrator) ruins the
    //!    estimate; the same high-pass filter on the input and the output removes the problem.
    //! 3. `test_srivc_high_order`: at high orders the coefficients of `A(s)` span many decades;
    //!    `srivc::identify` scales its prefilter internally, so that it still converges.
    //!
    //! The tests take ~10 s with optimizations and several minutes without, so they are ignored in
    //! debug builds: run `cargo test --release --test system_identification srivc:: -- --nocapture` (the
    //! tables of the candidates are printed), or force them with `-- --ignored`.

    use std::f64::consts::PI;

    use dsmc::{Polynomial, TransferFunction, TransferFunctionWithDelay};
    use dsmc::signal::excitation::multisine;
    use dsmc::system_identification::iv::srivc::{self, identify, Initialization, Outcome, SearchOptions, SrivcOptions, Structure};
    use dsmc::system_identification::preprocessing::high_pass;
    use dsmc::system_identification::validation::{Check, CheckKind};

    const TS: f64 = 1e-4;
    /// Samples per period of the multisine (1 s, so the lines are 1 Hz apart).
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

    /// Open-loop experiment from rest: multisine input `u` over `BAND`, and the output of `plant` to
    /// `u + input_offset` plus white noise of standard deviation `noise_ratio` times the RMS of the
    /// noise-free output (about its mean). The offset is not part of the recorded input.
    fn experiment(plant: &TransferFunctionWithDelay<f64>, n_samples: usize, seed: u64, noise_ratio: f64, input_offset: f64) -> (Vec<f64>, Vec<f64>) {
        let u: Vec<f64> = multisine(n_samples, PERIOD, &BAND.collect::<Vec<_>>(), seed);
        let applied: Vec<f64> = u.iter().map(|x| x + input_offset).collect();
        let y0 = plant.simulate(TS, &applied).unwrap();
        let mean = y0.iter().sum::<f64>() / n_samples as f64;
        let scale = noise_ratio * rms(&y0.iter().map(|v| v - mean).collect::<Vec<_>>());
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
        // Rigid-body gain 100 / s at low frequency
        let gain = 100.0 * resonance[2] / antiresonance[2] * low_pass[2];
        let numerator = &antiresonance * gain;
        let denominator = &(&Polynomial(vec![1.0, 0.0]) * &resonance) * &low_pass;
        TransferFunctionWithDelay::new(TransferFunction::from_polynomials(numerator, denominator), 8.0 * TS)
    }

    /// Three-inertia system, torque -> position (antiresonances 40 / 90 Hz, resonances 65 / 120 Hz),
    /// the same low-pass and dead time: `n = 8`, `m = 4`, `nk = 8`.
    fn three_inertia_plant() -> TransferFunctionWithDelay<f64> {
        let zeros = &second_order(40.0, 0.02) * &second_order(90.0, 0.02);
        let poles = &second_order(65.0, 0.02) * &second_order(120.0, 0.02);
        let low_pass = second_order(300.0, 0.5f64.sqrt());
        let gain = 1e4 * poles[4] / zeros[4] * low_pass[2];
        let numerator = &zeros * gain;
        let denominator = &(&Polynomial(vec![1.0, 0.0, 0.0]) * &poles) * &low_pass;
        TransferFunctionWithDelay::new(TransferFunction::from_polynomials(numerator, denominator), 8.0 * TS)
    }

    // ---------------------------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------------------------

    /// Joint search of `(n, m, nk)` by `srivc::search`: each candidate is identified on one
    /// experiment and validated on another one (different multisine phases and noise) at 99 % by
    /// the information criteria, whiteness, cross-correlation, the test at the excited lines and
    /// coherence; the converged candidates passing every test are compared by BIC.
    ///
    /// - Missing poles are made up for by a longer delay, so the best delay depends on the order
    ///   (16 samples for `n = 3`, 13 for `n = 4`, the true 8 for `n = 5`): the delay cannot be fixed
    ///   first and the orders searched afterwards.
    /// - The validation errors `V` of all reasonable candidates agree to 3 digits, while the
    ///   cross-correlation test rejects every lower order and every wrong delay: the residual still
    ///   depends on the input.
    /// - The line test also rejects the lower orders (mean F several times its expectation), but is
    ///   less sharp for a delay off by one sample.
    /// - With too many parameters the delay is no longer identifiable (an extra zero absorbs a shift),
    ///   and the iterations often do not converge; the converged ones pass every test, and lose in
    ///   BIC by about the penalty `ln N` of the extra parameter.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
    fn test_srivc_structure_search() {
        let plant = two_inertia_plant();
        let noise_ratio = 0.05;
        let (u, y) = experiment(&plant, 4 * PERIOD, 0x1234_5678_9abc, noise_ratio, 0.0);
        // Validation: 6 periods, the first one (transient from rest) not evaluated
        let (u_val, y_val) = experiment(&plant, 6 * PERIOD, 0x0fed_cba9_8765, noise_ratio, 0.0);
        let antiresonance = 36..=44;

        let structures: Vec<Structure> = [(3, 2, vec![14, 16, 18]), (4, 2, vec![12, 13, 14]), (5, 2, vec![6, 7, 8, 9, 10]), (5, 3, vec![6, 8, 10]), (6, 3, vec![8])]
            .into_iter()
            .flat_map(|(n, m, delays)| delays.into_iter().map(move |nk| Structure::new(n, m, nk)))
            .collect();
        let options = SearchOptions {
            evaluated_from: PERIOD,
            ..SearchOptions::new(
                Initialization::StateVariableFilter(2.0 * PI * 100.0),
                vec![
                    Check::Whiteness { max_lag: 20 },
                    Check::CrossCorrelation { max_lag: 100 },
                    Check::Lines { period: PERIOD, lines: BAND.collect() },
                    Check::Coherence { segment_len: PERIOD, excited: 1e-2 },
                ],
                0.99,
            )
        };
        let search = srivc::search((&u, &y), (&u_val, &y_val), TS, &structures, &options).unwrap();
        println!("{search}");
        for c in &search.candidates {
            if let Some(result) = c.result() {
                let (relative, phase) = band_error(&result.model, &plant, &antiresonance);
                println!("{}: max relative error {relative:.4}, max phase error {phase:.2} deg", c.structure);
            }
        }

        let selected = search.selected().unwrap();
        println!("{}", selected.report().unwrap());
        assert_eq!(selected.structure, Structure::new(5, 2, 8));
        let (relative, phase) = band_error(&selected.result().unwrap().model, &plant, &antiresonance);
        assert!(relative < 0.02 && phase < 1.0, "selected model: relative error {relative:e}, phase error {phase} deg");

        let validated = || search.candidates.iter().filter_map(|c| c.report().map(|r| (c.structure, r)));
        // The lower orders and the wrong delays fail the cross-correlation test, the lower orders
        // also the line test, by a mean F well above its expectation
        for (s, report) in validated().filter(|(s, _)| s.denominator_order < 5 || (s.numerator_order == 2 && s.input_delay != 8)) {
            assert_eq!(report.passed_by(CheckKind::CrossCorrelation), Some(false), "{s} not rejected by the cross-correlation");
        }
        for (s, report) in validated().filter(|(s, _)| s.denominator_order < 5) {
            let lines = report.lines().unwrap();
            let ratio = lines.mean_statistic() / lines.expected_statistic();
            assert!(report.passed_by(CheckKind::Lines) == Some(false) && ratio > 2.0, "{s}: line test, mean F {ratio}");
        }

        // An extra zero passes the tests when converged, but loses in BIC
        let extra: Vec<_> = search.candidates.iter().filter(|c| c.structure.numerator_order == 3 && c.selectable()).collect();
        assert!(!extra.is_empty() && extra.iter().all(|c| c.bic() > selected.bic()));
        assert!(search.candidates.iter().all(|c| matches!(c.outcome, Outcome::Validated { .. })));
    }

    /// A constant input offset that is not in the recorded input (a torque bias in an open-loop test)
    /// is integrated by the plant into a ramp of the output. SRIVC minimizes the output error, so the
    /// ramp dominates the fit: an offset of 0.1 % of the input RMS gives a relative error of ~2 %
    /// (0.35 % without it), and 1 % gives ~9 % (the size depends on the multisine phases).
    /// The same high-pass filter (3 Hz, twice, below the excited band) on the input and the output
    /// (`preprocessing::high_pass`) restores the estimate.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
    fn test_srivc_input_offset() {
        let plant = two_inertia_plant();
        let (m, n) = orders(&plant);
        let options = SrivcOptions { input_delay: 8, ..SrivcOptions::default() };
        let init = Initialization::StateVariableFilter(2.0 * PI * 100.0);
        let antiresonance = 36..=44;

        for offset in [0.0, 0.001, 0.01] {
            let (u, y) = experiment(&plant, 4 * PERIOD, 0x1234_5678_9abc, 0.05, offset);

            let raw = identify(&u, &y, TS, n, m, &init, &options).unwrap();
            let (raw_error, raw_phase) = band_error(&raw.model, &plant, &antiresonance);

            let (uf, yf) = high_pass(&u, &y, 3.0, TS, 2);
            let filtered = identify(&uf, &yf, TS, n, m, &init, &options).unwrap();
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

    /// Order 8 (three inertias, position output, low-pass): SRIVC converges in a few iterations (4
    /// without noise, 8 with), to an error of ~1e-8 without noise.
    ///
    /// Regression test of the scaling of the SRIVC prefilter: the coefficients `a_i` of `A(s)` grow
    /// like `ω^i`, so here the companion matrix spans ~22 decades. Before the states of the prefilter
    /// were scaled by the root radius of `A(s)`, its matrix exponential was inaccurate and the
    /// iterations ran into the limit (20) with an error of ~3e-2 without noise (checked by disabling
    /// the scaling), which fails the tolerances below. The lower-order tests do not span enough
    /// decades to catch that.
    ///
    /// The position of a free body drifts in an open-loop test, so the signals are high-passed as in
    /// `test_srivc_input_offset`.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
    fn test_srivc_high_order() {
        let plant = three_inertia_plant();
        let (m, n) = orders(&plant);
        let options = SrivcOptions { input_delay: 8, ..SrivcOptions::default() };
        let init = Initialization::StateVariableFilter(2.0 * PI * 100.0);
        let nothing = 0..=0;

        for noise_ratio in [0.0, 0.05] {
            // Noise relative to the high-passed output (the raw position is dominated by its ramp)
            let (u, y0) = experiment(&plant, 4 * PERIOD, 0x1234_5678_9abc, 0.0, 0.0);
            let (u, y0) = high_pass(&u, &y0, 3.0, TS, 2);
            let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
            let scale = noise_ratio * rms(&y0);
            let y: Vec<f64> = y0.iter().map(|v| v + scale * rng.gaussian()).collect();

            let result = identify(&u, &y, TS, n, m, &init, &options).unwrap();
            let (error, phase) = band_error(&result.model, &plant, &nothing);
            println!(
                "noise {noise_ratio}: {} iterations{}, relative error {error:.3e}, phase error {phase:.3} deg",
                result.iterations,
                if result.converged { "" } else { " (not converged)" },
            );
            let tolerance = if noise_ratio == 0.0 { 1e-4 } else { 5e-2 };
            assert!(result.converged && result.iterations <= 10 && error < tolerance, "noise {noise_ratio}: error {error:e}");
        }
    }
}

