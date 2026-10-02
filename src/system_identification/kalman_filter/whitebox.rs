//! Recursive (Kalman filter) estimation of a linear regression `y = φ^T θ` (white-box identification).

use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::Float;

/// Recursive estimation of `θ` in the linear regression `y = φ^T θ` by a Kalman filter,
/// modeling `θ` as a random walk (process noise variance `sigma_v`, measurement noise variance `sigma_w`),
/// so the estimate can follow slowly varying parameters. The regressor `φ` is built from a physical
/// model, e.g. `φ = [acceleration, velocity, sign(velocity)]`.
pub struct KalmanFilter<T> {
    /// Current estimate of `θ`.
    pub parameter: DVector<T>,
    covariance: DMatrix<T>,
    sigma_v: DMatrix<T>,
    sigma_w: T,
}

impl<T: Float + Default + AddAssign + SubAssign + MulAssign + DivAssign + Scalar> KalmanFilter<T> {
    /// `order`: number of parameters (length of `φ`). `sigma_v`: variance of the parameter random walk,
    /// `sigma_w`: variance of the measurement noise, `cov_0`: initial covariance of the parameters.
    pub fn new(order: usize, sigma_v: T, sigma_w: T, cov_0: T) -> Self {
        Self {
            parameter: DVector::zeros(order),
            covariance: DMatrix::identity(order, order) * cov_0,
            sigma_v: DMatrix::identity(order, order) * sigma_v,
            sigma_w,
        }
    }

    /// Update with regressor `phi` and output `y`.
    pub fn update(&mut self, phi: &[T], y: T) {
        let phi: DVector<T> = DVector::from_column_slice(phi);

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
}
