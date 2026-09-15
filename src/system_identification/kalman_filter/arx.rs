use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::Float;

use crate::TransferFunction;

pub struct KalmanFilter<T>
{
    pub u: DVector<T>,
    pub x: DVector<T>,
    pub parameter: DVector<T>,
    covariance: DMatrix<T>,
    sigma_v: DMatrix<T>,
    sigma_w: T,
    input_order: usize,
    state_order: usize,
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
        }
    }

    pub fn update(&mut self, u: T, x: T, y: T) {
        //FIFO for input u
        for i in (1..(self.input_order + 1)).rev() {
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
            phi[i + self.state_order] = self.u[i]
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

    pub fn identify(&self) -> TransferFunction<T> {
        let mut denom = Vec::<T>::with_capacity(self.state_order + 1);
        let mut numer = Vec::<T>::with_capacity(self.input_order + 1);

        denom.push(T::one());
        for i in 0..self.state_order {
            denom.push(-self.parameter[i]);
        }
        for i in 0..(self.input_order + 1) {
            numer.push(self.parameter[i + self.state_order]);
        }

        TransferFunction::new(&numer, &denom)
    }

}
