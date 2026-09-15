use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::Float;

pub struct KalmanFilter<T> {
    pub parameter: DVector<T>,
    covariance: DMatrix<T>,
    sigma_v: DMatrix<T>,
    sigma_w: T,
}

impl<T: Float + Default + AddAssign + SubAssign + MulAssign + DivAssign + Scalar> KalmanFilter<T> {
    pub fn new(order: usize, sigma_v: T, sigma_w: T, cov_0: T) -> Self {
        Self {
            parameter: DVector::zeros(order),
            covariance: DMatrix::identity(order, order) * cov_0,
            sigma_v: DMatrix::identity(order, order) * sigma_v,
            sigma_w,
        }
    }

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
