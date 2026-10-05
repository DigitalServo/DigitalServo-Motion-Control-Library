//! Gaussian process regression of a scalar function `y = f(x)`.

use std::ops::{AddAssign, MulAssign};

use num_traits::Float;

/// Gaussian process regression of a scalar function `y = f(x)` with a user-given kernel, from
/// samples with white measurement noise of variance `σ`:
///
/// ```text
/// mean  = kᵀ (K + σI)^-1 y
/// stdev = sqrt(k(x, x) - kᵀ (K + σI)^-1 k + σ)
/// ```
///
/// (`K_ij = k(x_i, x_j)`, `k_i = k(x_i, x)`). With `σ > 0` the mean smooths the noisy samples instead
/// of interpolating them, and repeated or close inputs do not make the covariance matrix singular.
/// Samples are added with `add`; the inverse covariance matrix is recomputed lazily in `predict`.
/// A typical kernel is the Gaussian kernel `k(x1, x2) = a exp(-(x1 - x2)^2 / (2 l^2))`.
pub struct GaussianProcessRegression<T> {
    /// Sampled inputs.
    pub x_sample: Vec<T>,
    /// Sampled outputs.
    pub y_sample: Vec<T>,
    /// Largest sampled input.
    pub x_max: T,
    /// Smallest sampled input.
    pub x_min: T,
    kernel: fn(T, T) -> T,
    sense_variance: T,
    inv_cov: Vec<Vec<T>>,
    sample: usize,
}

/// Prediction of `GaussianProcessRegression::predict`.
#[derive(Debug)]
pub struct PredictedValue<T> {
    /// Posterior mean.
    pub mean: T,
    /// Posterior standard deviation of a new measurement `y` (including the measurement noise).
    pub stdev: T,
}

impl<T: Float + AddAssign + MulAssign> GaussianProcessRegression<T> {
    /// `kernel(x1, x2)`: covariance function, `sigma`: variance of the measurement noise.
    pub fn new(kernel: fn(T, T) -> T, sigma: T) -> Self {
        Self {
            x_sample: vec![],
            y_sample: vec![],
            x_max: T::zero(),
            x_min: T::zero(),
            kernel,
            sense_variance: sigma,
            inv_cov: vec![vec![]],
            sample: 0,
        }
    }

    /// Add a sample `y = f(x)`.
    pub fn add(&mut self, x: T, y: T) {
        self.x_sample.push(x);
        self.y_sample.push(y);

        if self.sample == 0 {
            self.x_max = x;
            self.x_min = x;
        } else {
            if self.x_max < x {
                self.x_max = x;
            }
            if self.x_min > x {
                self.x_min = x;
            }
        }

        self.sample += 1;
    }

    /// Posterior mean of `f(x)` and standard deviation of a measurement `y` at `x`.
    ///
    /// Panics if `K + σI` is singular (only possible for `σ = 0`).
    pub fn predict(&mut self, x: T) -> PredictedValue<T> {
        if self.inv_cov.len() != self.sample {
            // K + σI
            let buffer: Vec<Vec<T>> = self.x_sample
                .iter()
                .enumerate()
                .map(|(i, &xi)| {
                    self.x_sample
                        .iter()
                        .enumerate()
                        .map(|(j, &xj)| (self.kernel)(xi, xj) + if i == j { self.sense_variance } else { T::zero() })
                        .collect()
                })
                .collect();
            self.inv_cov = inverse(&buffer).unwrap();
        }

        let k: Vec<T> = self.x_sample.iter().map(|&xi| (self.kernel)(xi, x)).collect();

        // kᵀ K^-1 v
        let quadratic = |v: &[T]| {
            self.inv_cov.iter().zip(&k).fold(T::zero(), |acc, (row, &ki)| {
                acc + ki * row.iter().zip(v).fold(T::zero(), |acc, (&a, &b)| acc + a * b)
            })
        };
        let mean: T = quadratic(&self.y_sample);
        let buffer2: T = quadratic(&k);

        let stdev: T = ((self.kernel)(x, x) - buffer2 + self.sense_variance)
            .abs()
            .sqrt();

        PredictedValue { mean, stdev }
    }
}

fn inverse<T: Float + MulAssign>(m: &[Vec<T>]) -> Option<Vec<Vec<T>>> {
    let vlen: usize = m.len();

    let mut m1: Vec<Vec<T>> = m.to_vec();
    let mut m2: Vec<Vec<T>> = vec![vec![T::zero(); vlen]; vlen];
    for (i, row) in m2.iter_mut().enumerate() {
        row[i] = T::one();
    }

    for i in 0..vlen {
        let mut max_row_option: usize = 0;
        let mut max_value_option: T = T::zero();
        for (j, row) in m1.iter().enumerate().skip(i) {
            if row[i].abs() > max_value_option {
                max_value_option = row[i].abs();
                max_row_option = j;
            }
        }

        /* Error Handling */
        if max_value_option == T::zero() {
            return None;
        }

        let m1_buffer: Vec<T> = m1[i].clone();
        let m2_buffer: Vec<T> = m2[i].clone();
        m1[i] = m1[max_row_option].clone();
        m2[i] = m2[max_row_option].clone();
        m1[max_row_option] = m1_buffer;
        m2[max_row_option] = m2_buffer;

        let scaler: T = T::one() / m1[i][i];
        for j in 0..vlen {
            m1[i][j] *= scaler;
            m2[i][j] *= scaler;
        }

        for j in 0..vlen {
            if i != j {
                let scaler: T = m1[j][i];
                for k in 0..vlen {
                    m1[j][k] = m1[j][k] - scaler * m1[i][k];
                    m2[j][k] = m2[j][k] - scaler * m2[i][k];
                }
            }
        }
    }

    Some(m2)
}
