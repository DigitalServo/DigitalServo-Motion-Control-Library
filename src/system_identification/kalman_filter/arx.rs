//! Recursive (Kalman filter) identification of an ARX model.

use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use nalgebra::{DMatrix, Scalar};
use num_traits::Float;

use crate::{Discrete, TransferFunction};
use crate::system_identification::arx::Arx;

/// Recursive identification of the ARX model
/// `y[k] = Σ_{i=1..na} a_i y[k-i] + Σ_{i=0..nb} b_i u[k-nk-i]`
/// (`na = state_order`, `nb = input_order`, `nk = input_delay`, 0 unless set by `with_input_delay`).
/// Call `update` every sample with `u[k]`, `y[k-1]` and `y[k]`, then `identify`.
pub struct KalmanFilter<T>
{
    /// Model with the current estimate `arx.parameter`.
    pub arx: Arx<T>,
    covariance: DMatrix<T>,
    sigma_v: DMatrix<T>,
    sigma_w: T,
}

impl<T: Float + AddAssign + SubAssign + MulAssign + DivAssign + Scalar> KalmanFilter<T>
{
    /// `state_order`: `na`, `input_order`: `nb`, `sigma_v`: variance of the parameter random walk,
    /// `sigma_w`: variance of the measurement noise, `cov_0`: initial covariance of the parameters.
    ///
    /// Argument order: the output (denominator) order `na` first, then the input (numerator)
    /// order `nb`, as in the notation `(na, nb, nk)`; e.g. the model
    /// `y[k] = a_1 y[k-1] + a_2 y[k-2] + b_0 u[k] + b_1 u[k-1]` is `(2, 1)`.
    pub fn new(state_order: usize, input_order: usize, sigma_v: T, sigma_w: T, cov_0: T) -> Self {
        Self::from_arx(Arx::new(state_order, input_order), sigma_v, sigma_w, cov_0)
    }

    /// Kalman filter of the given model structure (see `new` for the other arguments).
    pub fn from_arx(arx: Arx<T>, sigma_v: T, sigma_w: T, cov_0: T) -> Self {
        let n = arx.parameter_len();
        Self {
            arx,
            covariance: DMatrix::identity(n, n) * cov_0,
            sigma_v: DMatrix::identity(n, n) * sigma_v,
            sigma_w,
        }
    }

    /// Input delay `nk` \[samples\]: the model uses `u[k-nk] .. u[k-nk-nb]`, and `identify` gives
    /// `z^-nk B(z) / A(z)`. Set it before updating.
    pub fn with_input_delay(mut self, input_delay: usize) -> Self {
        self.arx = self.arx.with_input_delay(input_delay);
        self
    }

    /// Update with input `u = u[k]`, previous output `x = y[k-1]`, and output `y = y[k]`.
    pub fn update(&mut self, u: T, x: T, y: T) {
        self.arx.push(u, x);
        let phi = self.arx.regressor();

        let y_est: T = phi.dot(&self.arx.parameter);
        let y_err: T = y - y_est;

        //Predict step
        self.covariance += &self.sigma_v;

        //Update step
        let uncertainty_sense: T = self.sigma_w;
        let uncertainty_predict: T = phi.dot(&(&self.covariance * &phi));
        let uncertainty_observe: T = uncertainty_sense + uncertainty_predict;

        let x = &self.covariance * &phi;
        self.arx.parameter += (&x * y_err) / uncertainty_observe;
        self.covariance -= (&x * &x.transpose()) / uncertainty_observe;
    }

    /// `G(z)` of the current estimate.
    pub fn identify(&self) -> TransferFunction<T, Discrete> {
        self.arx.transfer_function()
    }

}
