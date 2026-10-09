//! Gaussian process regression.

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
