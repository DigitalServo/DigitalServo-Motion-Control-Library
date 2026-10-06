use dsmc::{TransferFunction, discretize::exact_discretize};

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
    let mut lsm = lsm::arx::DataBuffer::<f64>::new(input_order, state_order).with_input_delay(1);
    let mut kf = kalman_filter::arx::KalmanFilter::<f64>::new(input_order, state_order, 0.01, 0.01, 1.0e10).with_input_delay(1);

    let mut system = exact_discretize::DiscretizedSystem::from_tf(&tf_c, ts).unwrap();

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
    let tf_z = exact_discretize::discretize(&tf_c, ts).unwrap();

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

    let ret = levy::sanathanan_koerner_identification(&samples, numer_order, denom_order, 5).unwrap();

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
        let mut lsm = lsm::arx::DataBuffer::<f64>::new(b.len() - 1, a.len());
        let mut kf = kalman_filter::arx::KalmanFilter::<f64>::new(b.len() - 1, a.len(), 0.01, 0.01, 1.0e10);

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
    use dsmc::discretize::bilinear_transform;
    use dsmc::discretize::matched_z_transform::{
        to_continuous_with_delay, to_discrete_with_delay, ContinuousWithDelay, ToContinuousOptions, ZerosAtInfinity,
    };
    use dsmc::system_identification::{kalman_filter, lsm};

    let ts: f64 = 1e-3;
    let delay_samples = 5;
    let tf_c = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let plant = ContinuousWithDelay { tf: tf_c.clone(), delay: delay_samples as f64 * ts };
    // z^-5 (b0 z^2 + b1 z + b2) / (z^2 + a1 z + a2): na = nb = 2, nk = 5
    let tf_z = to_discrete_with_delay(&plant, ts, ZerosAtInfinity::MinusOne).unwrap();

    let mut system = bilinear_transform::DiscretizedSystem::from_tf_z(&tf_z);
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
    let structure = Arx::<f64>::new(1, 2).with_input_delay(1);

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
        let mut system = exact_discretize::DiscretizedSystem::from_tf(&tf_c, ts).unwrap();
        let u: Vec<f64> = (0..n_samples).map(|_| rand()).collect();
        let y0: Vec<f64> = u.iter().map(|&uk| system.update(&[uk]).unwrap()[0]).collect();

        let result = identify(&u, &y0, ts, m, n, &init, &options).unwrap();
        let err = relative_error(&result.model, &tf_c);
        assert!(result.converged && err < 1e-8, "noise-free: {} ({} iterations), error {err:e}", result.model, result.iterations);

        // Noise with a standard deviation of ~2.3 times the RMS of y0
        let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
        let y: Vec<f64> = y0.iter().map(|&v| v + 4.0 * rms * rand()).collect();
        let ls = identify(&u, &y, ts, m, n, &init, &SrivcOptions { max_iterations: 0, ..options.clone() }).unwrap();
        let err_ls = relative_error(&ls.model, &tf_c);
        let result = identify(&u, &y, ts, m, n, &init, &options).unwrap();
        let err_iv = relative_error(&result.model, &tf_c);
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
    let mut system = exact_discretize::DiscretizedSystem::from_tf(&tf_c, ts).unwrap();
    let u: Vec<f64> = (0..n_samples).map(|i| multisine(i as f64 * ts)).collect();
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let init = Initialization::Model(TransferFunction::continuous(&[500.0], &[1.0, 50.0, 700.0]));
    let options = SrivcOptions { input_delay: delay, ..options };

    let result = identify(&u, &y0, ts, 1, 2, &init, &options).unwrap();
    let err = relative_error(&result.model, &tf_c);
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

    let result = identify(&u, &y, ts, 1, 2, &init, &options).unwrap();
    let err = relative_error(&result.model, &tf_c);
    println!("Noisy ({noise_ratio} RMS): {:.3} ({} iterations), error {err:e}", result.model, result.iterations);
    assert!(result.converged && err < 1e-1, "with delay and noise: {} ({} iterations), error {err:e}", result.model, result.iterations);
}

/// `Validation`: simulation of continuous / discrete models against the measured output, mean
/// squared residual and BIC.
#[test]
fn test_validation_bic() {
    use dsmc::system_identification::validation::{Validation, ValidationError};
    use dsmc::Discrete;
    use std::f64::consts::PI;

    // Uniform (0, 1) and standard normal pseudo-random numbers (xorshift64, Box-Muller)
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 10000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let u: Vec<f64> = (0..n_samples).map(|_| gaussian()).collect();
    let mut system = exact_discretize::DiscretizedSystem::from_tf(&plant, ts).unwrap();
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();

    // The true model reproduces the noise-free output
    let validation = Validation::continuous(&plant, delay, ts, &u, &y0).unwrap();
    let max_residual = validation.residual.iter().fold(0.0, |acc: f64, v| acc.max(v.abs()));
    assert!(max_residual < 1e-10, "continuous: residual {max_residual:e}");

    // With white noise, V is the noise variance and BIC = N ln V + p ln N
    let sigma = 0.01;
    let y: Vec<f64> = y0.iter().map(|v| v + sigma * gaussian()).collect();
    let validation = Validation::continuous(&plant, delay, ts, &u, &y).unwrap();
    let v = validation.mse();
    assert!((v / (sigma * sigma) - 1.0).abs() < 0.05, "mse {v:e}");
    let n = n_samples as f64;
    assert!((validation.bic(3) - (n * v.ln() + 3.0 * n.ln())).abs() < 1e-9);

    // A wrong delay or a missing pole is penalized
    let wrong_delay = Validation::continuous(&plant, delay + 2, ts, &u, &y).unwrap();
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let wrong_order = Validation::continuous(&first_order, delay, ts, &u, &y).unwrap();
    println!("BIC: true {:.1}, wrong delay {:.1}, first order {:.1}", validation.bic(3), wrong_delay.bic(3), wrong_order.bic(2));
    assert!(wrong_delay.bic(3) > validation.bic(3) + 10.0 * n.ln());
    assert!(wrong_order.bic(2) > validation.bic(3) + 10.0 * n.ln());

    // Evaluation on the second half only
    let second_half = validation.clone().evaluated_from(n_samples / 2);
    assert_eq!(second_half.samples(), n_samples / 2);
    assert!((second_half.mse() / (sigma * sigma) - 1.0).abs() < 0.1);

    // Discrete-time model: y[k] = 1.5 y[k-1] - 0.7 y[k-2] + u[k-1] + 0.5 u[k-2]
    let g_z = TransferFunction::<f64, Discrete>::discrete(&[1.0, 0.5], &[1.0, -1.5, 0.7]);
    let mut y_arx = vec![0.0; n_samples];
    for k in 0..n_samples {
        let past = |x: &[f64], i: usize| if k >= i { x[k - i] } else { 0.0 };
        y_arx[k] = 1.5 * past(&y_arx, 1) - 0.7 * past(&y_arx, 2) + past(&u, 1) + 0.5 * past(&u, 2);
    }
    let validation = Validation::discrete(&g_z, &u, &y_arx).unwrap();
    let max_residual = validation.residual.iter().fold(0.0, |acc: f64, v| acc.max(v.abs()));
    assert!(max_residual < 1e-10, "discrete: residual {max_residual:e}");

    // Errors
    let improper = TransferFunction::continuous(&[1.0, 0.0, 0.0], &[1.0, 1.0]);
    assert!(matches!(Validation::continuous(&improper, 0, ts, &u, &y), Err(ValidationError::Improper { .. })));
    assert!(matches!(Validation::continuous(&plant, 0, ts, &u, &y[1..]), Err(ValidationError::LengthMismatch { .. })));
}

/// Cross-correlation test of the residual and the input: within the bound for the true model, far
/// outside for a wrong delay or a missing pole, with a white and with a colored (multisine) input.
#[test]
fn test_validation_cross_correlation() {
    use dsmc::system_identification::validation::Validation;
    use std::f64::consts::PI;

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 20000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let (max_lag, z) = (50, 2.58);

    let white: Vec<f64> = (0..n_samples).map(|_| gaussian()).collect();
    let phases: Vec<f64> = (0..40).map(|_| 2.0 * PI * uniform()).collect();
    let multisine: Vec<f64> = (0..n_samples)
        .map(|k| phases.iter().enumerate().map(|(i, p)| (2.0 * PI * (i + 1) as f64 * k as f64 * ts + p).sin()).sum::<f64>())
        .collect();

    for (name, u) in [("white", white), ("multisine", multisine)] {
        let mut system = exact_discretize::DiscretizedSystem::from_tf(&plant, ts).unwrap();
        let y0: Vec<f64> = (0..n_samples)
            .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
            .collect();
        let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
        let y: Vec<f64> = y0.iter().map(|v| v + 0.1 * rms * gaussian()).collect();

        let right = Validation::continuous(&plant, delay, ts, &u, &y).unwrap().cross_correlation(max_lag, z);
        let wrong_delay = Validation::continuous(&plant, delay + 1, ts, &u, &y).unwrap().cross_correlation(max_lag, z);
        let wrong_order = Validation::continuous(&first_order, delay, ts, &u, &y).unwrap().cross_correlation(max_lag, z);
        for (model, c) in [("true", &right), ("wrong delay", &wrong_delay), ("first order", &wrong_order)] {
            println!("{name} input, {model}: bound {:.4}, outside {:.1} %, max ratio {:.2}", c.bound, 100.0 * c.fraction_outside(), c.max_ratio());
        }
        // 99 % per lag: a few percent outside at most for the true model
        assert!(right.fraction_outside() < 0.05 && right.max_ratio() < 1.5, "{name}: true model {:?}", right.outside());
        assert!(wrong_delay.max_ratio() > 2.0 && wrong_delay.fraction_outside() > 0.1, "{name}: wrong delay");
        assert!(wrong_order.max_ratio() > 2.0 && wrong_order.fraction_outside() > 0.1, "{name}: first order");
    }
}

/// Line test with a periodic multisine: the true model is within the bound with white and with
/// colored noise (the noise level is estimated per line), wrong models are far outside.
#[test]
fn test_validation_line_test() {
    use dsmc::system_identification::validation::{Validation, ValidationError};
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let period = 1000; // 1 s: lines at 1 Hz spacing
    let lines: Vec<usize> = (1..=40).collect();
    let n_samples = 6 * period; // the first period (transient) is not evaluated: P = 5
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);

    let phases: Vec<f64> = lines.iter().map(|_| 2.0 * PI * uniform()).collect();
    let u: Vec<f64> = (0..n_samples)
        .map(|k| lines.iter().zip(&phases).map(|(&l, p)| (2.0 * PI * l as f64 * k as f64 / period as f64 + p).sin()).sum::<f64>())
        .collect();
    let mut system = exact_discretize::DiscretizedSystem::from_tf(&plant, ts).unwrap();
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();

    // White noise, and noise of the same power low-passed at ~5 Hz (concentrated at the low lines)
    let white: Vec<f64> = (0..n_samples).map(|_| 0.1 * rms * gaussian()).collect();
    let mut colored = vec![0.0; n_samples];
    for k in 1..n_samples {
        colored[k] = 0.97 * colored[k - 1] + (1.0 - 0.97f64 * 0.97).sqrt() * 0.1 * rms * gaussian();
    }

    for (name, noise) in [("white", &white), ("colored", &colored)] {
        let y: Vec<f64> = y0.iter().zip(noise.iter()).map(|(a, b)| a + b).collect();
        let test = |model: &TransferFunction<f64>, nk: usize| {
            Validation::continuous(model, nk, ts, &u, &y).unwrap().evaluated_from(period).line_test(period, &lines, 0.99).unwrap()
        };
        let (right, wrong_delay, wrong_order) = (test(&plant, delay), test(&plant, delay + 1), test(&first_order, delay));
        for (model, t) in [("true", &right), ("wrong delay", &wrong_delay), ("first order", &wrong_order)] {
            println!(
                "{name} noise, {model}: P = {}, bound {:.2}, outside {:.1} %, mean F {:.2} (expected {:.2}), relative error {:.4}",
                t.periods, t.bound, 100.0 * t.fraction_outside(), t.mean_statistic(), t.expected_statistic(), t.rms_relative_error()
            );
        }
        assert_eq!(right.periods, 5);
        assert!(right.fraction_outside() <= 0.05 && right.mean_statistic() < 2.0 * right.expected_statistic(), "{name}: true model");
        // One sample of delay matters at 5 .. 20 Hz, where the colored noise is ~3 times the white
        // one: there the shift is within the noise, and the test rightly does not reject it.
        if name == "white" {
            assert!(wrong_delay.fraction_outside() > 0.15 && wrong_delay.mean_statistic() > 3.0 * right.expected_statistic(), "{name}: wrong delay");
        }
        assert!(wrong_order.fraction_outside() > 0.5 && wrong_order.mean_statistic() > 10.0 * right.expected_statistic(), "{name}: first order");
    }

    // Errors
    let validation = Validation::continuous(&plant, delay, ts, &u, &y0).unwrap();
    assert!(matches!(validation.clone().evaluated_from(5 * period).line_test(period, &lines, 0.99), Err(ValidationError::TooFewPeriods { periods: 1 })));
    assert!(matches!(validation.line_test(period, &[600], 0.99), Err(ValidationError::InvalidLine { line: 600, .. })));
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

/// Coherence test of the residual and the input with non-periodic inputs (chirp, low-passed random
/// signal) and colored noise: the true model is within the bound at about the nominal rate, wrong
/// models are far outside.
#[test]
fn test_validation_coherence_test() {
    use dsmc::system_identification::validation::{Validation, ValidationError};
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 40000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let segment_len = 1000; // 1 Hz resolution

    // Linear chirp 0.5 .. 50 Hz over the record, and random noise low-passed at ~30 Hz
    let duration = n_samples as f64 * ts;
    let chirp: Vec<f64> = (0..n_samples)
        .map(|k| {
            let t = k as f64 * ts;
            (2.0 * PI * (0.5 * t + (50.0 - 0.5) * t * t / (2.0 * duration))).sin()
        })
        .collect();
    let mut random = vec![0.0; n_samples];
    for k in 1..n_samples {
        random[k] = 0.83 * random[k - 1] + gaussian();
    }

    for (name, u) in [("chirp", chirp), ("random", random)] {
        let mut system = exact_discretize::DiscretizedSystem::from_tf(&plant, ts).unwrap();
        let y0: Vec<f64> = (0..n_samples)
            .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
            .collect();
        let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
        // Colored noise (low-passed white noise), 10 % of the output RMS
        let mut noise = vec![0.0; n_samples];
        for k in 1..n_samples {
            noise[k] = 0.9 * noise[k - 1] + (1.0 - 0.81f64).sqrt() * 0.1 * rms * gaussian();
        }
        let y: Vec<f64> = y0.iter().zip(&noise).map(|(a, b)| a + b).collect();

        let test = |model: &TransferFunction<f64>, nk: usize, confidence: f64| {
            Validation::continuous(model, nk, ts, &u, &y).unwrap().coherence_test(segment_len, confidence).unwrap().excited(1e-2)
        };
        let (right, wrong_delay, wrong_order) = (test(&plant, delay, 0.99), test(&plant, delay + 1, 0.99), test(&first_order, delay, 0.99));
        for (model, t) in [("true", &right), ("wrong delay", &wrong_delay), ("first order", &wrong_order)] {
            println!(
                "{name}, {model}: K = {}, L = {:.1}, {} bins, bound {:.4}, outside {:.1} %, mean γ² {:.4} (expected {:.4})",
                t.segments, t.effective_segments, t.bins.len(), t.bound, 100.0 * t.fraction_outside(), t.mean_coherence(), t.expected_coherence()
            );
        }
        println!("{name}, wrong delay: outside at {:?} Hz", wrong_delay.outside());
        assert!(right.fraction_outside() < 0.05 && right.mean_coherence() < 1.5 * right.expected_coherence(), "{name}: true model");
        // One sample of delay shows where the plant has gain (the resonance at 5 Hz and around);
        // the random input also excites up to ~300 Hz, where it is buried in the noise, so the
        // fraction is taken up to 50 Hz
        let low = wrong_delay.outside().iter().filter(|&&f| f <= 50).count() as f64
            / wrong_delay.bins.iter().filter(|&&f| f <= 50).count() as f64;
        assert!(low > 0.15 && wrong_delay.outside().iter().filter(|&&f| (3..=11).contains(&f)).count() >= 8, "{name}: wrong delay");
        assert!(wrong_order.fraction_outside() > 0.3 && wrong_order.mean_coherence() > 5.0 * right.expected_coherence(), "{name}: first order");

        // Calibration of the bound for the true model: about 5 % of the bins outside at 95 %
        let at_95 = test(&plant, delay, 0.95);
        println!("{name}, true at 95 %: outside {:.1} %", 100.0 * at_95.fraction_outside());
        assert!(at_95.fraction_outside() < 0.12, "{name}: {:.3} outside at 95 %", at_95.fraction_outside());
    }

    // Errors
    let u = vec![0.0; 1400]; // one segment of 1000 (the next would start at 500)
    let validation = Validation::continuous(&plant, 0, ts, &u, &u).unwrap();
    assert!(matches!(validation.coherence_test(1000, 0.99), Err(ValidationError::TooFewSegments { segments: 1, .. })));
}

/// Frequency response of the model against the nonparametric estimate (random input, colored
/// noise): the error is at the noise level for the true model, the phase error of a wrong delay is
/// `-ω ts`, and a missing pole is far beyond the noise.
#[test]
fn test_validation_frequency_response() {
    use dsmc::system_identification::validation::Validation;
    use nalgebra::Complex;
    use std::f64::consts::PI;

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 40000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let segment_len = 1000;

    let mut u = vec![0.0; n_samples];
    for k in 1..n_samples {
        u[k] = 0.83 * u[k - 1] + gaussian();
    }
    let mut system = exact_discretize::DiscretizedSystem::from_tf(&plant, ts).unwrap();
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
    let mut noise = vec![0.0; n_samples];
    for k in 1..n_samples {
        noise[k] = 0.9 * noise[k - 1] + (1.0 - 0.81f64).sqrt() * 0.1 * rms * gaussian();
    }
    let y: Vec<f64> = y0.iter().zip(&noise).map(|(a, b)| a + b).collect();

    let compare = |model: &TransferFunction<f64>, nk: usize| {
        Validation::continuous(model, nk, ts, &u, &y).unwrap().frequency_response(segment_len).unwrap().excited(1e-2)
    };
    let (right, wrong_delay, wrong_order) = (compare(&plant, delay), compare(&plant, delay + 1), compare(&first_order, delay));
    for (name, c) in [("true", &right), ("wrong delay", &wrong_delay), ("first order", &wrong_order)] {
        println!(
            "{name}: {} bins, L = {:.1}, rms relative error {:.4}, mean normalized error {:.2}",
            c.bins.len(), c.effective_segments, c.rms_relative_error(), c.mean_normalized_error()
        );
    }

    // True model: the error is at the noise level, and small where the data are accurate
    // (output-input coherence above 0.9)
    let mean = right.mean_normalized_error();
    assert!(mean > 0.5 && mean < 2.0, "true model: mean normalized error {mean}");
    let accurate: Vec<usize> = (0..right.bins.len()).filter(|&i| right.coherence[i] > 0.9).collect();
    println!("accurate data at {:?} Hz", accurate.iter().map(|&i| right.bins[i]).collect::<Vec<_>>());
    assert!(accurate.len() >= 10);
    for &i in &accurate {
        assert!(right.gain_error_db()[i].abs() < 0.5 && right.phase_error()[i].abs() < 0.05, "{} Hz: true model", right.bins[i]);
    }

    // The model response is a windowed estimate: compared with the exact response of the sampled
    // system (the ZOH-discretized G(z) at z = e^(jω ts), with the delay; not G(jω), which lacks the
    // half-sample delay of the hold), its bias falls as the segments grow against the impulse
    // response of the plant (~0.1 s here)
    let g_z = exact_discretize::discretize(&plant, ts).unwrap();
    let bias = |segment_len: usize| {
        let c = Validation::continuous(&plant, delay, ts, &u, &y).unwrap().frequency_response(segment_len).unwrap();
        c.frequencies(ts)
            .iter()
            .enumerate()
            .filter(|&(_, &f)| (2.0..=20.0).contains(&f))
            .map(|(i, &f)| {
                let z = Complex::new(0.0, 2.0 * PI * f * ts).exp();
                let eval = |c: &[f64]| c.iter().fold(Complex::new(0.0, 0.0), |acc, &x| acc * z + x);
                let exact = eval(&g_z.numerator) / eval(&g_z.denominator) * z.powi(-(delay as i32));
                (c.model[i] - exact).norm() / exact.norm()
            })
            .fold(0.0, f64::max)
    };
    let (coarse, fine) = (bias(1000), bias(4000));
    println!("bias of the model response over 2 .. 20 Hz: {coarse:.4} (1 s segments), {fine:.4} (4 s)");
    assert!(fine < 0.05 && fine < 0.5 * coarse, "bias {coarse} -> {fine}");

    // One sample of delay: phase error -ω ts, gain unchanged, where the data are accurate
    let frequencies = wrong_delay.frequencies(ts);
    for &i in &accurate {
        let f = frequencies[i];
        let expected = -2.0 * PI * f * ts;
        assert!((wrong_delay.phase_error()[i] - expected).abs() < 0.2 * expected.abs() + 0.01, "{f} Hz: phase error {}", wrong_delay.phase_error()[i]);
        assert!(wrong_delay.gain_error_db()[i].abs() < 0.5, "{f} Hz: gain error {}", wrong_delay.gain_error_db()[i]);
    }
    // ... where it is beyond the noise (over all bins it is diluted by the high frequencies, where
    // the plant has no gain). Around the resonance the leakage of the 1 s segments lowers the
    // coherence and so inflates σ: there the normalized error of the true model is below its
    // nominal mean 1 (conservative)
    let at_accurate = |c: &dsmc::system_identification::validation::FrequencyResponseComparison<f64>| {
        let e = c.normalized_error();
        accurate.iter().map(|&i| e[i]).sum::<f64>() / accurate.len() as f64
    };
    println!("mean normalized error where accurate: true {:.2}, wrong delay {:.2}", at_accurate(&right), at_accurate(&wrong_delay));
    assert!(at_accurate(&wrong_delay) > 1.0 && at_accurate(&wrong_delay) > 5.0 * at_accurate(&right));

    // A missing pole is far beyond the uncertainty of the data
    assert!(wrong_order.mean_normalized_error() > 100.0 && wrong_order.rms_relative_error() > 0.5);
}

/// AIC, AICc and BIC: definitions, and their penalties against each other.
#[test]
fn test_validation_aic() {
    use dsmc::system_identification::validation::Validation;

    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let u: Vec<f64> = (0..1000).map(|k| ((k * 7919) % 1000) as f64 / 500.0 - 1.0).collect();
    let y: Vec<f64> = (0..1000).map(|k| 0.01 * (((k * 104729) % 997) as f64 / 498.5 - 1.0)).collect();
    let validation = Validation::continuous(&plant, 0, 1e-3, &u, &y).unwrap();
    let (n, v) = (1000.0, validation.mse());

    assert!((validation.aic(3) - (n * v.ln() + 6.0)).abs() < 1e-9);
    assert!((validation.aicc(3) - (validation.aic(3) + 24.0 / 996.0)).abs() < 1e-9);
    assert!((validation.bic(3) - (n * v.ln() + 3.0 * n.ln())).abs() < 1e-9);
    // One more parameter costs 2 in AIC and ln N ≈ 6.9 in BIC
    assert!((validation.aic(4) - validation.aic(3) - 2.0).abs() < 1e-9);
    assert!((validation.bic(4) - validation.bic(3) - n.ln()).abs() < 1e-9);
    // AICc diverges when the parameters approach the samples
    let short = validation.clone().evaluated_from(995);
    assert!(short.aicc(3).is_finite() && short.aicc(4).is_infinite());
}

/// Whiteness test of the residual: a white residual passes, a colored one fails. For an output
/// error model with colored noise, the true model fails the whiteness test (the noise is colored)
/// but passes the cross-correlation test (`G` is right).
#[test]
fn test_validation_whiteness() {
    use dsmc::system_identification::validation::Validation;
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let n_samples = 20000;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let first_order = TransferFunction::continuous(&[50.0], &[1.0, 50.0]);
    let (max_lag, z) = (20, 2.58);
    let z_one_sided = 2.326; // 99 %

    let u: Vec<f64> = (0..n_samples).map(|_| gaussian()).collect();
    let mut system = exact_discretize::DiscretizedSystem::from_tf(&plant, ts).unwrap();
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();
    let white: Vec<f64> = (0..n_samples).map(|_| 0.1 * rms * gaussian()).collect();
    let mut colored = vec![0.0; n_samples];
    for k in 1..n_samples {
        colored[k] = 0.9 * colored[k - 1] + (1.0 - 0.81f64).sqrt() * 0.1 * rms * gaussian();
    }

    let check = |model: &TransferFunction<f64>, noise: &[f64]| {
        let y: Vec<f64> = y0.iter().zip(noise).map(|(a, b)| a + b).collect();
        let validation = Validation::continuous(model, delay, ts, &u, &y).unwrap();
        (validation.autocorrelation(max_lag, z), validation.cross_correlation(50, z))
    };

    // Wilson-Hilferty against the exact 99 % quantile of χ²(20), 37.566
    let (white_true, white_cross) = check(&plant, &white);
    assert!((white_true.ljung_box_bound(z_one_sided) - 37.566).abs() < 0.2, "{}", white_true.ljung_box_bound(z_one_sided));

    let (colored_true, colored_cross) = check(&plant, &colored);
    let (white_wrong, _) = check(&first_order, &white);
    for (name, w) in [("true, white noise", &white_true), ("true, colored noise", &colored_true), ("first order, white noise", &white_wrong)] {
        println!(
            "{name}: outside {:.0} %, max ratio {:.2}, Ljung-Box {:.1} (bound {:.1})",
            100.0 * w.fraction_outside(), w.max_ratio(), w.ljung_box, w.ljung_box_bound(z_one_sided)
        );
    }

    // White noise, true model: white
    assert!(white_true.fraction_outside() <= 0.1 && white_true.ljung_box <= white_true.ljung_box_bound(z_one_sided));
    assert!(white_cross.fraction_outside() < 0.05);
    // Colored noise, true model: not white, but independent of the input
    assert!(colored_true.max_ratio() > 10.0 && colored_true.ljung_box > 10.0 * colored_true.ljung_box_bound(z_one_sided));
    assert!(colored_cross.fraction_outside() < 0.05, "colored noise: cross-correlation {:?}", colored_cross.outside());
    // Wrong model: the unmodeled response (G - Ĝ) u makes the residual colored
    assert!(white_wrong.ljung_box > 10.0 * white_wrong.ljung_box_bound(z_one_sided));
}

/// One-step prediction error of ARX models: white for data with a white equation error (the ARX
/// noise model holds), colored for white output noise even with the true parameters, where the
/// IV estimate validated by its output error passes instead.
#[test]
fn test_validation_one_step_prediction() {
    use dsmc::system_identification::{arx::Arx, iv, lsm, validation::Validation};
    use nalgebra::DVector;
    use std::f64::consts::PI;

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    // y[k] = 1.5 y[k-1] - 0.7 y[k-2] + u[k-1] + 0.5 u[k-2] (+ noise): na = 2, nb = 1, nk = 1
    let n_samples = 20000;
    let structure = Arx::<f64>::new(1, 2).with_input_delay(1);
    let mut truth = structure.clone();
    truth.parameter = DVector::from_vec(vec![1.5, -0.7, 1.0, 0.5]);
    let simulate = |u: &[f64], e: &[f64]| {
        let mut y = vec![0.0; u.len()];
        for k in 0..u.len() {
            let past = |x: &[f64], i: usize| if k >= i { x[k - i] } else { 0.0 };
            y[k] = 1.5 * past(&y, 1) - 0.7 * past(&y, 2) + past(u, 1) + 0.5 * past(u, 2) + e[k];
        }
        y
    };
    let least_squares = |structure: &Arx<f64>, u: &[f64], y: &[f64]| {
        let mut buffer = lsm::arx::DataBuffer::from_arx(structure.clone());
        for k in 0..u.len() {
            buffer.add(u[k], if k > 0 { y[k - 1] } else { 0.0 }, y[k]);
        }
        buffer.identify().unwrap();
        buffer.arx
    };
    let (max_lag, z, z_one_sided) = (20, 2.58, 2.326);
    let white = |v: &Validation<f64>| {
        let w = v.autocorrelation(max_lag, z);
        (w.ljung_box, w.ljung_box_bound(z_one_sided))
    };

    let u: Vec<f64> = (0..n_samples).map(|_| gaussian()).collect();
    let e: Vec<f64> = (0..n_samples).map(|_| 0.1 * gaussian()).collect();

    // Equation error: the prediction error of the true model is e itself
    let y = simulate(&u, &e);
    let validation = Validation::one_step_prediction(&truth, &u, &y).unwrap();
    let max_difference = validation.residual.iter().zip(&e).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(max_difference < 1e-12, "prediction error of the true model differs from e by {max_difference:e}");

    // ... and the least-squares estimate is white and independent of the input
    let estimate = Validation::one_step_prediction(&least_squares(&structure, &u, &y), &u, &y).unwrap();
    let (q, bound) = white(&estimate);
    println!("equation error, LS: Ljung-Box {q:.1} (bound {bound:.1}), mse {:.4e}", estimate.mse());
    assert!(q <= bound && estimate.cross_correlation(50, z).fraction_outside() < 0.05);
    assert!((estimate.mse() / 0.01 - 1.0).abs() < 0.05);

    // ... while a missing pole leaves a colored prediction error
    let low = Validation::one_step_prediction(&least_squares(&Arx::new(1, 1).with_input_delay(1), &u, &y), &u, &y).unwrap();
    let (q_low, _) = white(&low);
    println!("equation error, na = 1: Ljung-Box {q_low:.1}");
    assert!(q_low > 10.0 * bound);

    // Output error y = G u + v: even the true A, B leave the colored error A(z) v
    let y0 = simulate(&u, &vec![0.0; n_samples]);
    let y: Vec<f64> = y0.iter().map(|v| v + 0.3 * gaussian()).collect();
    let (q_true, _) = white(&Validation::one_step_prediction(&truth, &u, &y).unwrap());
    let (q_ls, _) = white(&Validation::one_step_prediction(&least_squares(&structure, &u, &y), &u, &y).unwrap());
    // ... the IV estimate is validated by its output error instead: white, as v
    let iv_model = iv::arx::identify(&u, &y, &structure, 3).unwrap();
    let output_error = Validation::discrete(&iv_model.transfer_function(), &u, &y).unwrap();
    let (q_iv, _) = white(&output_error);
    println!("output error: prediction error of the true model {q_true:.1}, of LS {q_ls:.1}; output error of IV {q_iv:.1}");
    assert!(q_true > 10.0 * bound && q_ls > 10.0 * bound);
    assert!(q_iv <= bound && output_error.cross_correlation(50, z).fraction_outside() < 0.05);
}


// ---------------------------------------------------------------------------------------------
// SRIVC on multi-inertia plants
// ---------------------------------------------------------------------------------------------

mod srivc {
    //! SRIVC on plants typical of an open-loop motion-control identification test: multi-inertia
    //! mechanics (rigid-body integrator, lightly damped resonances / antiresonances) seen through an
    //! analog low-pass filter and a dead time, excited by a band-limited multisine.
    //!
    //! Three things are checked, each of them needed to use `srivc::identify` on such data:
    //!
    //! 1. `test_srivc_structure_search`: the denominator order `n`, the numerator order `m` and the
    //!    input delay `nk` have to be searched *jointly*, and validated by the residual tests of
    //!    `Validation` (not by the raw validation error, which differs only in the 4th digit between
    //!    candidates), then compared by BIC.
    //! 2. `test_srivc_input_offset`: a small constant input offset (drift of the integrator) ruins the
    //!    estimate; the same high-pass filter on the input and the output removes the problem.
    //! 3. `test_srivc_time_normalization`: at high orders the coefficients of `A(s)` span many
    //!    decades; `srivc::identify` scales the prefilter internally, so the plain call converges and
    //!    agrees with an explicit time normalization (`identify_normalized`).
    //!
    //! The tests take ~10 s with optimizations and several minutes without, so they are ignored in
    //! debug builds: run `cargo test --release --test system_identification srivc:: -- --nocapture` (the
    //! tables of the candidates are printed), or force them with `-- --ignored`.

    use std::f64::consts::PI;

    use dsmc::TransferFunction;
    use dsmc::discretize::exact_discretize::DiscretizedSystem;
    use dsmc::system_identification::iv::srivc::{Initialization, SrivcOptions, identify};
    use dsmc::system_identification::validation::Validation;
    use num_complex::Complex;

    const TS: f64 = 1e-4;
    /// Excited band of the multisine \[Hz\] (1 Hz spacing, so the period is 1 s).
    const BAND: std::ops::RangeInclusive<usize> = 10..=200;

    // ---------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------

    /// xorshift64 pseudo-random numbers.
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

    /// Product of two polynomials (coefficients in descending order).
    fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                c[i + j] += x * y;
            }
        }
        c
    }

    /// `s^2 + 2 ζ ω s + ω^2` with `ω = 2π f`.
    fn second_order(f: f64, zeta: f64) -> Vec<f64> {
        let w = 2.0 * PI * f;
        vec![1.0, 2.0 * zeta * w, w * w]
    }

    fn rms(v: &[f64]) -> f64 {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }

    /// Continuous-time model `e^(-delay ts s) B(s) / A(s)` (coefficients in descending order).
    #[derive(Clone, Debug)]
    struct Model {
        numerator: Vec<f64>,
        denominator: Vec<f64>,
        /// Input delay \[samples\].
        delay: usize,
    }

    impl Model {
        /// `G(jω)` including the dead time.
        fn response(&self, omega: f64) -> Complex<f64> {
            let s = Complex::new(0.0, omega);
            let eval = |c: &[f64]| c.iter().fold(Complex::new(0.0, 0.0), |acc, &x| acc * s + x);
            eval(&self.numerator) / eval(&self.denominator) * Complex::new(0.0, -omega * self.delay as f64 * TS).exp()
        }

        /// Sampled response to the input `u` applied through a zero-order hold, from rest.
        /// `None` if the model cannot be realized or the response diverges (unstable model).
        fn simulate(&self, u: &[f64]) -> Option<Vec<f64>> {
            let tf = TransferFunction::continuous(&self.numerator, &self.denominator);
            let mut system = DiscretizedSystem::from_tf(&tf, TS).ok()?;
            let y: Vec<f64> = (0..u.len())
                .map(|k| system.update(&[if k >= self.delay { u[k - self.delay] } else { 0.0 }]).unwrap()[0])
                .collect();
            y.iter().all(|v| v.is_finite()).then_some(y)
        }
    }

    /// Multisine over `BAND` with random phases and unit RMS, starting at `k = 0` (zero before).
    fn multisine(n_samples: usize, seed: u64) -> Vec<f64> {
        let mut rng = Xorshift(seed);
        let phases: Vec<f64> = BAND.map(|_| 2.0 * PI * rng.uniform()).collect();
        let scale = (2.0 / phases.len() as f64).sqrt();
        (0..n_samples)
            .map(|k| {
                let t = k as f64 * TS;
                scale * BAND.zip(&phases).map(|(f, p)| (2.0 * PI * f as f64 * t + p).sin()).sum::<f64>()
            })
            .collect()
    }

    /// Open-loop experiment from rest: multisine input `u`, and the output of `plant` to
    /// `u + input_offset` plus white noise of standard deviation `noise_ratio` times the RMS of the
    /// noise-free output (about its mean). The offset is not part of the recorded input.
    fn experiment(plant: &Model, n_samples: usize, seed: u64, noise_ratio: f64, input_offset: f64) -> (Vec<f64>, Vec<f64>) {
        let u = multisine(n_samples, seed);
        let applied: Vec<f64> = u.iter().map(|x| x + input_offset).collect();
        let y0 = plant.simulate(&applied).unwrap();
        let mean = y0.iter().sum::<f64>() / n_samples as f64;
        let scale = noise_ratio * rms(&y0.iter().map(|v| v - mean).collect::<Vec<_>>());
        let mut rng = Xorshift(seed ^ 0x0abc_def1_2345);
        let y = y0.iter().map(|v| v + scale * rng.gaussian()).collect();
        (u, y)
    }

    /// First-order high-pass filter with cutoff `fc` \[Hz\], at rest before `k = 0`.
    ///
    /// To remove a drift it has to be applied to the input *and* the output: the plant is linear, so
    /// the filtered signals are still related by the same `G(s)`, and a filtered sampled input is
    /// still constant between samples, so the zero-order-hold assumption of SRIVC stays exact.
    ///
    /// The first sample matters: `x[-1] = 0` here, i.e. the step of the signal at `k = 0` is filtered
    /// too. Starting from `x[-1] = x[0]` instead breaks the relation between the two filtered signals
    /// (an error of 5 .. 20 % on the plant below).
    fn high_pass(x: &[f64], fc: f64) -> Vec<f64> {
        let a = 1.0 / (1.0 + 2.0 * PI * fc * TS);
        let mut out = vec![0.0; x.len()];
        let (mut previous_in, mut previous_out) = (0.0, 0.0);
        for (k, &v) in x.iter().enumerate() {
            out[k] = a * (previous_out + v - previous_in);
            previous_in = v;
            previous_out = out[k];
        }
        out
    }

    /// Result of `identify_normalized`.
    struct Fit {
        model: Model,
        iterations: usize,
        converged: bool,
    }

    /// SRIVC in normalized time `t' = w0 t` (`s' = s / w0`), `w0` \[rad/s\] being a frequency inside
    /// the band of interest; `w0 = 1` is the plain call. The coefficients of `A(s)` grow like
    /// `ω^i` and span many decades at high orders; `identify` scales its prefilter by the root radius
    /// of `A(s)` itself, so the normalization is not needed for convergence (see
    /// `test_srivc_time_normalization`) and only changes the rounding.
    ///
    /// The fitted `G'(s') = Σ b'_j s'^(m-j) / (s'^n + Σ a'_i s'^(n-i))` is `G(s) = G'(s / w0)`:
    /// `a_i = a'_i w0^i`, `b_j = b'_j w0^(n-m+j)`.
    ///
    /// `svf` is the bandwidth \[rad/s\] of the state-variable filter starting the iterations.
    fn identify_normalized(u: &[f64], y: &[f64], m: usize, n: usize, delay: usize, svf: f64, w0: f64) -> Option<Fit> {
        let options = SrivcOptions { input_delay: delay, max_iterations: 20, ..SrivcOptions::default() };
        let result = identify(u, y, TS * w0, m, n, &Initialization::StateVariableFilter(svf / w0), &options).ok()?;
        let theta = result.parameter.as_slice();
        if theta.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let mut denominator = vec![1.0];
        denominator.extend((1..=n).map(|i| theta[i - 1] * w0.powi(i as i32)));
        let numerator = (0..=m).map(|j| theta[n + j] * w0.powi((n - m + j) as i32)).collect();
        Some(Fit { model: Model { numerator, denominator, delay }, iterations: result.iterations, converged: result.converged })
    }

    /// Model error against the true plant over `BAND`, leaving out `exclude` \[Hz\] (the notch of an
    /// antiresonance, where the relative error is dominated by the tiny gain):
    /// (max relative error `|G - G0| / |G0|`, max phase error \[deg\]).
    fn band_error(model: &Model, plant: &Model, exclude: &std::ops::RangeInclusive<usize>) -> (f64, f64) {
        BAND.filter(|f| !exclude.contains(f)).fold((0.0, 0.0), |(relative, phase): (f64, f64), f| {
            let omega = 2.0 * PI * f as f64;
            let (g, g0) = (model.response(omega), plant.response(omega));
            (relative.max((g - g0).norm() / g0.norm()), phase.max((g / g0).arg().abs().to_degrees()))
        })
    }

    // ---------------------------------------------------------------------------------------------
    // Plants
    // ---------------------------------------------------------------------------------------------

    /// Two-inertia system, torque -> velocity (antiresonance 40 Hz, resonance 65 Hz, ζ = 0.02),
    /// second-order Butterworth low-pass at 300 Hz in the measurement, dead time of 8 samples (0.8 ms):
    /// `n = 5`, `m = 2`, `nk = 8`.
    fn two_inertia_plant() -> Model {
        let (antiresonance, resonance, low_pass) = (second_order(40.0, 0.02), second_order(65.0, 0.02), second_order(300.0, 0.5f64.sqrt()));
        // Rigid-body gain 100 / s at low frequency
        let gain = 100.0 * resonance[2] / antiresonance[2] * low_pass[2];
        Model {
            numerator: antiresonance.iter().map(|c| c * gain).collect(),
            denominator: convolve(&convolve(&[1.0, 0.0], &resonance), &low_pass),
            delay: 8,
        }
    }

    /// Three-inertia system, torque -> position (antiresonances 40 / 90 Hz, resonances 65 / 120 Hz),
    /// the same low-pass and dead time: `n = 8`, `m = 4`, `nk = 8`.
    fn three_inertia_plant() -> Model {
        let zeros = convolve(&second_order(40.0, 0.02), &second_order(90.0, 0.02));
        let poles = convolve(&second_order(65.0, 0.02), &second_order(120.0, 0.02));
        let low_pass = second_order(300.0, 0.5f64.sqrt());
        let gain = 1e4 * poles[4] / zeros[4] * low_pass[2];
        Model {
            numerator: zeros.iter().map(|c| c * gain).collect(),
            denominator: convolve(&convolve(&[1.0, 0.0, 0.0], &poles), &low_pass),
            delay: 8,
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------------------------

    /// Joint search of `(n, m, nk)`: each candidate is identified on one experiment and validated on
    /// another one (different multisine phases and noise) with `Validation`:
    ///
    /// - BIC `N ln V + p ln N` (`V`: mean squared simulation error, `p = n + m + 1`),
    /// - cross-correlation test of the residual and the input (`τ = -100 ..= 100`, 99 % per lag),
    /// - test at the excited lines of the multisine (model error against the noise level estimated
    ///   per line from the period-to-period variation, 99 % per line).
    ///
    /// The candidates passing both tests (at most 5 % of the lags / lines outside, for a nominal 1 %)
    /// are compared by BIC, and the lowest is selected.
    ///
    /// - Missing poles are made up for by a longer delay, so the best delay depends on the order
    ///   (16 samples for `n = 3`, 13 for `n = 4`, the true 8 for `n = 5`): the delay cannot be fixed
    ///   first and the orders searched afterwards.
    /// - The validation errors `V` of all reasonable candidates agree to 3 digits, while the
    ///   cross-correlation test rejects every lower order and every wrong delay (40 .. 80 % of the
    ///   lags outside): the residual still depends on the input.
    /// - The line test is less sharp with 5 periods at 99 % (a delay off by one sample passes with
    ///   1 .. 3 % of the lines outside, mean F 1.2 .. 1.3 times its expectation), but rejects the lower
    ///   orders by their mean F (2 .. 70 times its expectation).
    /// - With too many parameters the delay is no longer identifiable (an extra zero absorbs a shift),
    ///   and the iterations often do not converge; the converged ones pass both tests, and lose in BIC
    ///   by about the penalty `ln N` of the extra parameter.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
    fn test_srivc_structure_search() {
        let plant = two_inertia_plant();
        let period = (1.0 / TS).round() as usize;
        let noise_ratio = 0.05;
        let (u, y) = experiment(&plant, 4 * period, 0x1234_5678_9abc, noise_ratio, 0.0);
        // Validation: 6 periods, the first one (transient from rest) not used by the line test
        let (u_val, y_val) = experiment(&plant, 6 * period, 0x0fed_cba9_8765, noise_ratio, 0.0);
        let lines: Vec<usize> = BAND.collect();

        let w0 = 2.0 * PI * 60.0;
        let svf = 2.0 * PI * 100.0;
        let antiresonance = 36..=44;

        // (n, m, delays)
        let candidates: [(usize, usize, &[usize]); 5] =
            [
                (3, 2, &[14, 16, 18]),
                (4, 2, &[12, 13, 14]),
                (5, 2, &[6, 7, 8, 9, 10]),
                (5, 3, &[6, 8, 10]),
                (6, 3, &[8]),
            ];

        struct Row {
            n: usize,
            m: usize,
            fit: Fit,
            bic: f64,
            /// Cross-correlation test: fraction of the lags outside, max |r| / bound
            correlation: (f64, f64),
            /// Line test: fraction of the lines outside, mean F / expected F
            lines: (f64, f64),
        }
        impl Row {
            fn passes(&self) -> bool {
                self.fit.converged && self.correlation.0 <= 0.05 && self.lines.0 <= 0.05
            }
        }
        let mut rows = Vec::new();
        println!(" n  m nk |      BIC | corr. out  max |  lines out  mean F | max rel. err | max phase err | iterations");
        for (n, m, delays) in candidates {
            for &delay in delays {
                let Some(fit) = identify_normalized(&u, &y, m, n, delay, svf, w0) else { continue };
                let model = TransferFunction::continuous(&fit.model.numerator, &fit.model.denominator);
                let validation = Validation::continuous(&model, delay, TS, &u_val, &y_val).unwrap();
                if !validation.mse().is_finite() {
                    println!("{n:2} {m:2} {delay:2} | unstable model ({} iterations)", fit.iterations);
                    continue;
                }
                let bic = validation.bic(n + m + 1);
                let correlation = validation.cross_correlation(100, 2.58);
                let line_test = validation.clone().evaluated_from(period).line_test(period, &lines, 0.99).unwrap();
                let row = Row {
                    n,
                    m,
                    bic,
                    correlation: (correlation.fraction_outside(), correlation.max_ratio()),
                    lines: (line_test.fraction_outside(), line_test.mean_statistic() / line_test.expected_statistic()),
                    fit,
                };
                let (relative, phase) = band_error(&row.fit.model, &plant, &antiresonance);
                println!(
                    "{n:2} {m:2} {delay:2} | {bic:8.1} | {:7.1} % {:5.2} | {:7.1} % {:7.2} | {relative:12.4} | {phase:9.2} deg | {}{}{}",
                    100.0 * row.correlation.0,
                    row.correlation.1,
                    100.0 * row.lines.0,
                    row.lines.1,
                    row.fit.iterations,
                    if row.fit.converged { "" } else { " (not converged)" },
                    if row.passes() { "" } else { " rejected" },
                );
                rows.push(row);
            }
        }

        // Selection: the lowest BIC among the candidates passing both tests
        let selected = rows.iter().filter(|r| r.passes()).min_by(|a, b| a.bic.total_cmp(&b.bic)).unwrap();
        println!("selected: n = {}, m = {}, nk = {}", selected.n, selected.m, selected.fit.model.delay);
        assert_eq!((selected.n, selected.m, selected.fit.model.delay), (5, 2, 8));

        let (relative, phase) = band_error(&selected.fit.model, &plant, &antiresonance);
        assert!(relative < 0.02 && phase < 1.0, "selected model: relative error {relative:e}, phase error {phase} deg");

        // The lower orders and the wrong delays are rejected by the cross-correlation test, the lower
        // orders also by the mean F of the line test
        for r in rows.iter().filter(|r| r.n < 5 || (r.m == 2 && r.fit.model.delay != 8)) {
            assert!(r.correlation.0 > 0.3 && r.correlation.1 > 2.0, "({}, {}, {}) not rejected by the cross-correlation", r.n, r.m, r.fit.model.delay);
        }
        for r in rows.iter().filter(|r| r.n < 5) {
            assert!(r.lines.1 > 2.0, "({}, {}, {}): mean F {}", r.n, r.m, r.fit.model.delay, r.lines.1);
        }

        // An extra zero passes the tests when converged, but loses in BIC
        let extra: Vec<&Row> = rows.iter().filter(|r| r.m == 3 && r.passes()).collect();
        assert!(!extra.is_empty() && extra.iter().all(|r| r.bic > selected.bic));
    }

    /// A constant input offset that is not in the recorded input (a torque bias in an open-loop test)
    /// is integrated by the plant into a ramp of the output. SRIVC minimizes the output error, so the
    /// ramp dominates the fit: an offset of 0.1 % of the input RMS gives a relative error of ~3 %
    /// (0.35 % without it), and 1 % gives ~40 %.
    /// The same high-pass filter (3 Hz, twice, below the excited band) on the input and the output
    /// restores the estimate.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
    fn test_srivc_input_offset() {
        let plant = two_inertia_plant();
        let (m, n) = (plant.numerator.len() - 1, plant.denominator.len() - 1);
        let w0 = 2.0 * PI * 60.0;
        let svf = 2.0 * PI * 100.0;
        let antiresonance = 36..=44;
        let filter = |x: &[f64]| high_pass(&high_pass(x, 3.0), 3.0);

        for offset in [0.0, 0.001, 0.01] {
            let (u, y) = experiment(&plant, 40000, 0x1234_5678_9abc, 0.05, offset);

            let raw = identify_normalized(&u, &y, m, n, plant.delay, svf, w0).unwrap();
            let (raw_error, raw_phase) = band_error(&raw.model, &plant, &antiresonance);

            let filtered = identify_normalized(&filter(&u), &filter(&y), m, n, plant.delay, svf, w0).unwrap();
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

    /// Order 8 (three inertias, position output, low-pass): the plain call (`w0 = 1`) and the call in
    /// normalized time both converge in a few iterations (4 without noise, 6 with), to an error of
    /// ~1e-6 without noise, and agree. Before the prefilter of `identify` was scaled by the root radius
    /// of `A(s)`, the plain call ran into the iteration limit here (error ~5e-4 without noise): the
    /// matrix exponential of the unscaled companion matrix, whose entries span ~22 decades, was
    /// inaccurate.
    ///
    /// The position of a free body drifts in an open-loop test, so the signals are high-passed as in
    /// `test_srivc_input_offset`.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "slow without optimizations; run with `cargo test --release`")]
    fn test_srivc_time_normalization() {
        let plant = three_inertia_plant();
        let (m, n) = (plant.numerator.len() - 1, plant.denominator.len() - 1);
        let svf = 2.0 * PI * 100.0;
        let nothing = 0..=0;
        let filter = |x: &[f64]| high_pass(&high_pass(x, 3.0), 3.0);

        for noise_ratio in [0.0, 0.05] {
            // Noise relative to the high-passed output (the raw position is dominated by its ramp)
            let (u, y0) = experiment(&plant, 40000, 0x1234_5678_9abc, 0.0, 0.0);
            let (u, y0) = (filter(&u), filter(&y0));
            let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
            let scale = noise_ratio * rms(&y0);
            let y: Vec<f64> = y0.iter().map(|v| v + scale * rng.gaussian()).collect();

            let mut errors = Vec::new();
            for (name, w0) in [("plain", 1.0), ("normalized", 2.0 * PI * 60.0)] {
                let fit = identify_normalized(&u, &y, m, n, plant.delay, svf, w0).unwrap();
                let (error, phase) = band_error(&fit.model, &plant, &nothing);
                println!(
                    "noise {noise_ratio}: {name:10} {:2} iterations{}, relative error {error:.3e}, phase error {phase:.3} deg",
                    fit.iterations,
                    if fit.converged { "" } else { " (not converged)" },
                );
                let tolerance = if noise_ratio == 0.0 { 1e-4 } else { 5e-2 };
                assert!(fit.converged && fit.iterations <= 10 && error < tolerance, "noise {noise_ratio}: {name}, error {error:e}");
                errors.push(error);
            }
            // Same estimate up to rounding
            assert!((errors[0] - errors[1]).abs() < 1e-2 * errors[1].max(1e-6), "noise {noise_ratio}: {errors:?}");
        }
    }
}

/// `Validation::check`: several tests at one confidence level with pass / fail rules. The true
/// model passes, a wrong delay fails the input tests, colored noise fails the whiteness test only,
/// and the true model is rejected by chance at about the nominal rate.
#[test]
fn test_validation_check() {
    use dsmc::system_identification::validation::{Check, CheckResult, Validation};
    use std::f64::consts::PI;

    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();

    let ts: f64 = 1e-3;
    let period = 1000;
    let n_samples = 6 * period;
    let delay = 3;
    let plant = TransferFunction::continuous(&[1000.0], &[1.0, 20.0, 1000.0]);
    let lines: Vec<usize> = (1..=40).collect();
    // Periodic multisine (also used by the non-periodic tests)
    let phases: Vec<f64> = lines.iter().map(|_| 2.0 * PI * uniform()).collect();
    let u: Vec<f64> = (0..n_samples)
        .map(|k| lines.iter().zip(&phases).map(|(&l, p)| (2.0 * PI * l as f64 * k as f64 / period as f64 + p).sin()).sum::<f64>())
        .collect();
    let mut system = exact_discretize::DiscretizedSystem::from_tf(&plant, ts).unwrap();
    let y0: Vec<f64> = (0..n_samples)
        .map(|k| system.update(&[if k >= delay { u[k - delay] } else { 0.0 }]).unwrap()[0])
        .collect();
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n_samples as f64).sqrt();

    let checks = [
        Check::InformationCriteria { parameters: 3 },
        Check::Whiteness { max_lag: 20 },
        Check::CrossCorrelation { max_lag: 50 },
        Check::Coherence { segment_len: 500, excited: 1e-2 },
        Check::Lines { period, lines: lines.clone() },
    ];
    let mut noisy = |colored: bool| {
        let mut v = 0.0;
        y0.iter()
            .map(|y| {
                v = if colored { 0.9 * v + (1.0 - 0.81f64).sqrt() * 0.1 * rms * gaussian() } else { 0.1 * rms * gaussian() };
                y + v
            })
            .collect::<Vec<f64>>()
    };
    let run = |nk: usize, y: &[f64]| {
        Validation::continuous(&plant, nk, ts, &u, y).unwrap().evaluated_from(period).check(&checks, 0.99).unwrap()
    };
    let passed = |report: &dsmc::system_identification::validation::Report<f64>| -> Vec<Option<bool>> {
        report.results.iter().map(CheckResult::passed).collect()
    };

    let white = noisy(false);
    let right = run(delay, &white);
    println!("true model, white noise:\n{right}");
    assert!(right.passed() && passed(&right) == [None, Some(true), Some(true), Some(true), Some(true)]);

    let wrong = run(delay + 1, &white);
    println!("wrong delay:\n{wrong}");
    assert!(!wrong.passed());
    assert_eq!(passed(&wrong)[2..4], [Some(false), Some(false)], "cross-correlation and coherence");

    let colored = run(delay, &noisy(true));
    println!("true model, colored noise:\n{colored}");
    assert_eq!(passed(&colored), [None, Some(false), Some(true), Some(true), Some(true)]);

    // False rejections of the true model over independent noise records. Measured over 400
    // records (release build): whiteness 3, cross-correlation 0 (Bonferroni, conservative with
    // correlated lags; counting the lags outside against a binomial quantile instead gave 48),
    // coherence 1, lines 5, i.e. about the nominal 1 % per test.
    let trials = 40;
    let rejected = (0..trials).filter(|_| !run(delay, &noisy(false)).passed()).count();
    println!("true model rejected in {rejected} of {trials} trials (4 tests at 99 %)");
    assert!(rejected <= 6, "{rejected} of {trials}");
}
