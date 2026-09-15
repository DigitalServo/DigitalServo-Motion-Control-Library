use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::Float;

pub struct KalmanFilter<T> {
    pub parameter: DVector<T>,
    covariance: DMatrix<T>,
    sigma_v: DMatrix<T>,
    sigma_w: T,
    order: usize,
}

impl<T: Float + AddAssign + SubAssign + MulAssign + DivAssign + Scalar> KalmanFilter<T>
{
    pub fn new(order: usize, sigma_v: T, sigma_w: T, cov_0: T) -> Self {
        Self {
            parameter: DVector::zeros(order + 1),
            covariance: DMatrix::identity(order + 1, order + 1) * cov_0,
            sigma_v: DMatrix::identity(order + 1, order + 1) * sigma_v,
            sigma_w,
            order
        }
    }

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

    pub fn identify(&self) -> Vec<T> {
        self.parameter.as_slice().to_vec()
    }
}
