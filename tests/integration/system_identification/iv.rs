//! Instrumental variables for ARX models with colored noise.

use dsmc::TransferFunction;

use super::*;

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
