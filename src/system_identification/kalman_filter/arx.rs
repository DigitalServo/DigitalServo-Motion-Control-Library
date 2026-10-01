use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::Float;

use crate::{Discrete, TransferFunction};

/// Recursive identification of the ARX model
/// `y[k] = Σ_{i=1..na} a_i y[k-i] + Σ_{i=0..nb} b_i u[k-nk-i]`
/// (`na = state_order`, `nb = input_order`, `nk = input_delay`, 0 unless set by `with_input_delay`).
pub struct KalmanFilter<T>
{
    /// Input history `u[k], u[k-1], ..., u[k-nk-nb]`.
    pub u: DVector<T>,
    pub x: DVector<T>,
    pub parameter: DVector<T>,
    covariance: DMatrix<T>,
    sigma_v: DMatrix<T>,
    sigma_w: T,
    input_order: usize,
    state_order: usize,
    input_delay: usize,
}

impl<T: Float + AddAssign + SubAssign + MulAssign + DivAssign + Scalar> KalmanFilter<T>
{
    pub fn new(input_order: usize, state_order: usize, sigma_v: T, sigma_w: T, cov_0: T) -> Self {
        Self {
            u: DVector::zeros(input_order + 1),
            x: DVector::zeros(state_order),
            parameter: DVector::zeros(input_order + state_order + 1),
            covariance: DMatrix::identity(input_order + state_order + 1, input_order + state_order + 1) * cov_0,
            sigma_v: DMatrix::identity(input_order + state_order + 1, input_order + state_order + 1) * sigma_v,
            sigma_w,
            input_order,
            state_order,
            input_delay: 0,
        }
    }

    /// Input delay `nk` [samples]: the model uses `u[k-nk] .. u[k-nk-nb]`, and `identify` gives
    /// `z^-nk B(z) / A(z)`. Set it before updating.
    pub fn with_input_delay(mut self, input_delay: usize) -> Self {
        self.input_delay = input_delay;
        self.u = DVector::zeros(input_delay + self.input_order + 1);
        self
    }

    pub fn update(&mut self, u: T, x: T, y: T) {
        //FIFO for input u
        for i in (1..self.u.len()).rev() {
            self.u[i] = self.u[i - 1]
        }
        self.u[0] = u;

        //FIFO for state x
        for i in (1..self.state_order).rev() {
            self.x[i] = self.x[i - 1]
        }
        self.x[0] = x;

        let mut phi: DVector<T> = DVector::zeros(self.input_order + self.state_order + 1);
        for i in 0..self.state_order {
            phi[i] = self.x[i]
        }
        for i in 0..(self.input_order + 1) {
            phi[i + self.state_order] = self.u[i + self.input_delay]
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

    pub fn identify(&self) -> TransferFunction<T, Discrete> {
        let (a, b) = self.parameter.as_slice().split_at(self.state_order);
        crate::system_identification::arx::transfer_function(a, b, self.input_delay)
    }

}
