//! Least-squares estimation of a linear regression `y = φ^T θ` (white-box identification).

use std::ops::{AddAssign, MulAssign};
use nalgebra::{ComplexField, DMatrix, DVector, Scalar};
use num_traits::Float;

/// Least-squares estimation of `θ` in the linear regression `y = φ^T θ`.
/// Samples are accumulated with `add` (only `Σ φ φ^T` and `Σ φ y` are kept).
/// The regressor `φ` is built from a physical model, e.g. `φ = [acceleration, velocity, sign(velocity)]`
/// for inertia, viscous friction and Coulomb friction.
pub struct DataBuffer<T> {
    psi_sum: DVector<T>,
    phi_sum: DMatrix<T>,
}

impl<T: Float + AddAssign + MulAssign + ComplexField + Scalar> DataBuffer<T> {
    /// `order`: number of parameters (length of `φ`).
    pub fn new(order: usize) -> Self {
        Self {
            psi_sum: DVector::zeros(order),
            phi_sum: DMatrix::zeros(order, order),
        }
    }

    /// Add a sample with regressor `phi` and output `y`.
    pub fn add(&mut self, phi: &[T], y: T) {
        let phi = DVector::from_column_slice(phi);
        self.psi_sum += &phi * y;
        self.phi_sum += &phi * &phi.transpose();
    }

    /// Least-squares `θ`; `None` if the data do not determine it.
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
