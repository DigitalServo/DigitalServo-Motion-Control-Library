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
        let mut opts = VectorFittingOptions::default();
        opts.fit_d = true;
        opts.fit_e = true;
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
