//! Recursive (Kalman filter) polynomial fit.

use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::Float;

/// Recursive polynomial fit `y = Σ θ_i x^(order - i)` by a Kalman filter (see `whitebox::KalmanFilter`);
/// `identify` returns the descending-order coefficients.
pub struct KalmanFilter<T> {
    /// Current estimate of the descending-order coefficients.
    pub parameter: DVector<T>,
    covariance: DMatrix<T>,
    sigma_v: DMatrix<T>,
    sigma_w: T,
    order: usize,
}

impl<T: Float + AddAssign + SubAssign + MulAssign + DivAssign + Scalar> KalmanFilter<T>
{
    /// Polynomial of degree `order`. `sigma_v`: variance of the parameter random walk,
    /// `sigma_w`: variance of the measurement noise, `cov_0`: initial covariance of the parameters.
    pub fn new(order: usize, sigma_v: T, sigma_w: T, cov_0: T) -> Self {
        Self {
            parameter: DVector::zeros(order + 1),
            covariance: DMatrix::identity(order + 1, order + 1) * cov_0,
            sigma_v: DMatrix::identity(order + 1, order + 1) * sigma_v,
            sigma_w,
            order
        }
    }

    /// Update with a sample `y = f(x)`.
    pub fn update(&mut self, x: T, y: T) {
        let mut phi: DVector<T> = DVector::zeros(self.order + 1);
        for i in 0..(self.order + 1) {
            phi[i] = x.powi((self.order - i) as i32);
        }

        let y_est: T = phi.dot(&self.parameter);
        let y_err: T = y - y_est;

        //Predict step
        self.covariance += &self.sigma_v;

        //Update step
        let uncertainty_sense: T = self.sigma_w;
        let uncertainty_predict: T = phi.dot(&(&self.covariance * &phi));
        let uncertainty_observe: T = uncertainty_sense + uncertainty_predict;

        let x = &self.covariance * &phi;
        self.parameter += (&x * y_err) / uncertainty_observe;
        self.covariance -= (&x * &x.transpose()) / uncertainty_observe;
    }

    /// Descending-order coefficients of the current estimate.
    pub fn identify(&self) -> Vec<T> {
        self.parameter.as_slice().to_vec()
    }
}
