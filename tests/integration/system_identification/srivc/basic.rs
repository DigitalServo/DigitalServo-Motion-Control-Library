//! SRIVC on a simple plant: noise-free and noisy data (against the least-squares start), and a dead time.

use dsmc::{DiscreteSystem, StateSpace, TransferFunction, discretize::Zoh};

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
