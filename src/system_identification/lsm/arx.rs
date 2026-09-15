use std::ops::{AddAssign, MulAssign};
use nalgebra::{ComplexField, DMatrix, DVector};
use num_traits::Float;

use crate::TransferFunction;

//Use sequential data
pub struct DataBuffer<T> {
    pub u: DVector<T>,
    pub x: DVector<T>,
    psi_sum: DVector<T>,
    phi_sum: DMatrix<T>,
    input_order: usize,
    state_order: usize,
}

impl<T: Float + AddAssign + MulAssign + ComplexField> DataBuffer<T> {
    pub fn new(input_order: usize, state_order: usize) -> Self {
        Self {
            u: DVector::zeros(input_order + 1),
            x: DVector::zeros(state_order),
            psi_sum: DVector::zeros(input_order + state_order + 1),
            phi_sum: DMatrix::zeros(input_order + state_order + 1, input_order + state_order + 1),
            input_order,
            state_order,
        }
    }

    pub fn add(&mut self, u: T, x: T, y: T) {
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

        let mut phi = DVector::zeros(self.input_order + self.state_order + 1);
        for i in 0..self.state_order {
            phi[i] = self.x[i]
        }
        for i in 0..(self.input_order + 1) {
            phi[i + self.state_order] = self.u[i]
        }

        self.psi_sum += &phi * y;
        self.phi_sum += &phi * &phi.transpose();
    }

    pub fn identify(&self) -> Option<TransferFunction<T>> {
        match self.phi_sum.clone().try_inverse() {
            Some(res) => {
                let theta = &res * &self.psi_sum;

                let mut denom = Vec::<T>::with_capacity(self.state_order + 1);
                let mut numer = Vec::<T>::with_capacity(self.input_order + 1);

                denom.push(T::one());
                for i in 0..self.state_order {
                    denom.push(-theta[i]);
                }
                for i in 0..(self.input_order + 1) {
                    numer.push(theta[i + self.state_order]);
                }

                Some(TransferFunction::new(&numer, &denom))
            }
            None => None,
        }
    }
}
