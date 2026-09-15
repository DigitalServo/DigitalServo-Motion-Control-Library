use std::ops::{AddAssign, MulAssign};
use nalgebra::{ComplexField, DMatrix, DVector};
use num_traits::Float;

pub struct DataBuffer<T> {
    psi_sum: DVector<T>,
    phi_sum: DMatrix<T>,
    order: usize,
}

impl<T: Float + AddAssign + MulAssign + ComplexField> DataBuffer<T>
{
    pub fn new(order: usize) -> Self {
        Self {
            psi_sum: DVector::zeros(order + 1),
            phi_sum: DMatrix::zeros(order + 1, order + 1),
            order,
        }
    }

    pub fn add(&mut self, x: T, y: T) {
        let mut phi = DVector::zeros(self.order + 1);
        for i in 0..(self.order + 1) {
            phi[i] = <T as Float>::powi(x, (self.order - i) as i32);
        }

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
