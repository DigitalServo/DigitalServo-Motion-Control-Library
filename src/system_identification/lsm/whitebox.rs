use std::ops::{AddAssign, MulAssign};
use nalgebra::{ComplexField, DMatrix, DVector, Scalar};
use num_traits::Float;

//Use sequential data
pub struct DataBuffer<T> {
    psi_sum: DVector<T>,
    phi_sum: DMatrix<T>,
}

impl<T: Float + AddAssign + MulAssign + ComplexField + Scalar> DataBuffer<T> {
    pub fn new(order: usize) -> Self {
        Self {
            psi_sum: DVector::zeros(order),
            phi_sum: DMatrix::zeros(order, order),
        }
    }

    pub fn add(&mut self, phi: &[T], y: T) {
        let phi = DVector::from_column_slice(phi);
        self.psi_sum += &phi * y;
        self.phi_sum += &phi * &phi.transpose();
    }

    pub fn identify(&self) -> Option<Vec<T>> {
        match self.phi_sum.clone().try_inverse() {
            Some(res) => {
                let theta = &res * &self.psi_sum;
                Some(theta.data.as_vec().to_vec())
            }
            None => None,
        }
    }
}
