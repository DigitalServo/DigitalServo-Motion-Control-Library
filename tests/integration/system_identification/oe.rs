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
    assert_eq!(result.termination, oe::Termination::Converged, "after {} iterations", result.iterations);
    assert!(err < TOL_LSM_ARX, "OE (noise-free, from a model): error {err:e}");
    assert!(result.loss < 1e-20, "loss {:e}", result.loss);
}

/// Output-error data `y = G u + v` with white (uniform) measurement noise `v`: the OE model
/// minimizes the simulation error, below that of its IV initial model, and is closer to `G`.
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
    assert_eq!(result.termination, oe::Termination::Converged, "after {} iterations", result.iterations);
    assert!(err_oe < 0.02, "OE: error {err_oe:e}");
    assert!(err_oe < err_iv, "OE: error {err_oe:e}, IV {err_iv:e}");

    // The fixed point of SRIV is the stationary point of the loss: OE from it stops at once.
    let sriv = iv::arx::identify_sriv(&u, &y, &structure, 10).unwrap();
    let from_sriv = oe::identify(&u, &y, &structure, &oe::Initialization::Sriv(10), &options).unwrap();
    let gap = tf_distance(&sriv.transfer_function(), &result.model.transfer_function());
    assert!(gap < 1e-8, "SRIV vs OE: {gap:e}");
    assert_eq!(from_sriv.termination, oe::Termination::Converged);
    assert!(from_sriv.iterations <= 1, "OE from SRIV: {} iterations", from_sriv.iterations);
    assert!(tf_distance(&from_sriv.model.transfer_function(), &result.model.transfer_function()) < 1e-8);

    // The loss is the variance of the noise, (0.5 rms)^2 / 3, and below that of the IV model.
    let noise_var = (0.5 * rms).powi(2) / 3.0;
    let iv_result =
        oe::identify(&u, &y, &structure, &oe::Initialization::Model(iv_model), &oe::OeOptions { max_iterations: 0, ..options })
            .unwrap();
    assert_eq!(iv_result.termination, oe::Termination::MaxIterations);
    let iv_loss = iv_result.loss;
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

/// Lightly damped resonance sampled fast (`ω0 / 2π = 20 Hz`, `ζ = 0.02`, `ts = 1 ms`, the poles of
/// `F` at `|z| ≈ 0.997`): the plain IV estimate is unstable here, while SRIV starts the iterations
/// at the minimum of the loss.
#[test]
fn test_oe_lightly_damped() {
    use dsmc::discretize::Zoh;

    let (w, zeta, ts) = (2.0 * std::f64::consts::PI * 20.0, 0.02, 1e-3);
    let g = TransferFunction::continuous(&[w * w], &[1.0, 2.0 * zeta * w, w * w]).discretize(Zoh, ts).unwrap();
    let (num, den) = (&g.numerator, &g.denominator);
    let (b, f) = ([num[1] / den[0], num[2] / den[0]], [-den[1] / den[0], -den[2] / den[0]]);

    // Input held for 4 samples, noise with a standard deviation of ~17 % of that of y0
    let mut rand = rand_source(0x2545_f491_4f6c_dd1d ^ 0x9e37_79b9_7f4a_7c15);
    let n = 20000;
    let (mut u, mut y0) = (vec![0.0; n], vec![0.0; n]);
    for k in 0..n {
        u[k] = if k % 4 == 0 { rand() } else { u[k - 1] };
        let y = |i: usize| if k >= i { y0[k - i] } else { 0.0 };
        let u = |i: usize| if k >= i { u[k - i] } else { 0.0 };
        y0[k] = f[0] * y(1) + f[1] * y(2) + b[0] * u(1) + b[1] * u(2);
    }
    let rms = (y0.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    let y: Vec<f64> = y0.iter().map(|&v| v + 0.3 * rms * rand()).collect();
    let noise_var = (0.3 * rms).powi(2) / 3.0;
    let structure = Arx::<f64>::new(2, 1).with_input_delay(1);

    let result = oe::identify(&u, &y, &structure, &oe::Initialization::Sriv(10), &oe::OeOptions::default()).unwrap();
    let p = &result.model.parameter;
    println!("OE from SRIV: {:?} after {} iterations, F {:?}", result.termination, result.iterations, &p.as_slice()[..2]);
    assert_eq!(result.termination, oe::Termination::Converged);
    assert!((result.loss / noise_var - 1.0).abs() < 0.05, "OE loss {:e}, noise variance {noise_var:e}", result.loss);
    let err_f = (p[0] - f[0]).abs().max((p[1] - f[1]).abs());
    assert!(err_f < 1e-4, "F error {err_f:e}");
    // The DC gain B(1) / F(1) = 1 of the plant
    let gain = (p[2] + p[3]) / (1.0 - p[0] - p[1]);
    assert!((gain - 1.0).abs() < 0.02, "DC gain {gain}");

    // The unstable plain IV estimate is stabilized instead of rejected.
    let from_iv = oe::identify(&u, &y, &structure, &oe::Initialization::InstrumentalVariable(3), &oe::OeOptions::default()).unwrap();
    println!("OE from IV: {:?} after {} iterations, loss / σ² {}", from_iv.termination, from_iv.iterations, from_iv.loss / noise_var);
    assert!(from_iv.model.is_stable());

    // Least squares loses the resonance (two real poles): OE from it ends in a local minimum (still
    // ~30 σ² after the default 50 iterations; 10 keep the test short).
    let options = oe::OeOptions { max_iterations: 10, ..oe::OeOptions::default() };
    let from_ls = oe::identify(&u, &y, &structure, &oe::Initialization::LeastSquares, &options).unwrap();
    println!("OE from LS: {:?} after {} iterations, loss / σ² {}", from_ls.termination, from_ls.iterations, from_ls.loss / noise_var);
    assert!(from_ls.loss > 10.0 * noise_var, "OE from LS: loss / σ² {}", from_ls.loss / noise_var);
}
