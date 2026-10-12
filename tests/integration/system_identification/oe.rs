//! Output-error models by the prediction-error method.

use dsmc::system_identification::{arx::Arx, iv, oe};
use dsmc::{Discrete, TransferFunction};

use super::*;

/// Uniform pseudo-random numbers in [-1, 1) (xorshift64).
fn rand_source(mut state: u64) -> impl FnMut() -> f64 {
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// `y0[k] = 1.5 y0[k-1] - 0.7 y0[k-2] + 1.0 u[k-1] + 0.5 u[k-2]` (na = 2, nb = 1, nk = 1) driven by
/// white noise: the input, the noise-free output, the structure and `G(z)`.
fn plant(n: usize, rand: &mut impl FnMut() -> f64) -> (Vec<f64>, Vec<f64>, Arx<f64>, TransferFunction<f64, Discrete>) {
    let (a, b) = ([1.5, -0.7], [1.0, 0.5]);
    let (mut u, mut y0) = (vec![0.0; n], vec![0.0; n]);
    for k in 0..n {
        u[k] = rand();
        let y = |i: usize| if k >= i { y0[k - i] } else { 0.0 };
        let u = |i: usize| if k >= i { u[k - i] } else { 0.0 };
        y0[k] = a[0] * y(1) + a[1] * y(2) + b[0] * u(1) + b[1] * u(2);
    }
    let expected = TransferFunction::discrete(&[0.0, 1.0, 0.5], &[1.0, -1.5, 0.7]);
    (u, y0, Arx::new(2, 1).with_input_delay(1), expected)
}

#[test]
fn test_oe_noise_free() {
    let mut rand = rand_source(0x2545_f491_4f6c_dd1d);
    let (u, y0, structure, expected) = plant(2000, &mut rand);

    // From a poor (but stable) initial model, the iterations reach the plant exactly.
    let mut initial = structure.clone();
    initial.parameter.copy_from_slice(&[1.0, -0.3, 0.5, 0.0]);
    let result = oe::identify(&u, &y0, &structure, &oe::Initialization::Model(initial), &oe::OeOptions::default()).unwrap();
    let err = tf_distance(&result.model.transfer_function(), &expected);
    assert!(result.converged, "not converged after {} iterations", result.iterations);
    assert!(err < TOL_LSM_ARX, "OE (noise-free, from a model): error {err:e}");
    assert!(result.loss < 1e-20, "loss {:e}", result.loss);
}

/// Output-error data `y = G u + v` with white measurement noise `v`: the OE model is the
/// maximum-likelihood estimate, with a smaller simulation error than that of its IV initial model.
#[test]
fn test_oe_noisy() {
    let mut rand = rand_source(0x2545_f491_4f6c_dd1d);
    let n = 20000;
    let (u, y0, structure, expected) = plant(n, &mut rand);

    // Noise with a standard deviation of ~30 % of that of y0
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    let y: Vec<f64> = y0.iter().map(|&v| v + 0.5 * rms * rand()).collect();

    let iv_model = iv::arx::identify(&u, &y, &structure, 1).unwrap();
    let err_iv = tf_distance(&iv_model.transfer_function(), &expected);

    let options = oe::OeOptions::default();
    let result = oe::identify(&u, &y, &structure, &oe::Initialization::InstrumentalVariable(1), &options).unwrap();
    let err_oe = tf_distance(&result.model.transfer_function(), &expected);
    println!("IV error {err_iv:e}, OE error {err_oe:e} ({} iterations)", result.iterations);
    assert!(result.converged, "not converged after {} iterations", result.iterations);
    assert!(err_oe < 0.02, "OE: error {err_oe:e}");
    assert!(err_oe < err_iv, "OE: error {err_oe:e}, IV {err_iv:e}");

    // The loss is the variance of the noise, (0.5 rms)^2 / 3, and below that of the IV model.
    let noise_var = (0.5 * rms).powi(2) / 3.0;
    let iv_loss = oe::identify(&u, &y, &structure, &oe::Initialization::Model(iv_model), &oe::OeOptions { max_iterations: 0, ..options })
        .unwrap()
        .loss;
    assert!(result.loss <= iv_loss, "OE loss {:e}, IV loss {iv_loss:e}", result.loss);
    assert!((result.loss / noise_var - 1.0).abs() < 0.05, "OE loss {:e}, noise variance {noise_var:e}", result.loss);
}

#[test]
fn test_oe_errors() {
    let mut rand = rand_source(0x2545_f491_4f6c_dd1d);
    let (u, y0, structure, _) = plant(200, &mut rand);
    let options = oe::OeOptions::default();
    let init = oe::Initialization::InstrumentalVariable(1);

    assert_eq!(
        oe::identify(&u[..100], &y0, &structure, &init, &options).unwrap_err(),
        oe::OeError::LengthMismatch { input: 100, output: 200 }
    );
    assert!(matches!(
        oe::identify(&u, &y0, &structure, &init, &oe::OeOptions { evaluated_from: 198, ..options.clone() }),
        Err(oe::OeError::TooFewSamples { .. })
    ));

    let wrong = oe::Initialization::Model(Arx::new(2, 1));
    assert!(matches!(oe::identify(&u, &y0, &structure, &wrong, &options), Err(oe::OeError::InitialModel { .. })));

    // F(z) = 1 - 1.1 z^-1 - 0 z^-2: pole at z = 1.1
    let mut unstable = structure.clone();
    unstable.parameter.copy_from_slice(&[1.1, 0.0, 1.0, 0.5]);
    assert_eq!(
        oe::identify(&u, &y0, &structure, &oe::Initialization::Model(unstable), &options).unwrap_err(),
        oe::OeError::UnstableInitialModel
    );
}
