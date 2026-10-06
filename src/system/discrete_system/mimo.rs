//! `DiscreteSystem<T, Mimo>`: any number of inputs and outputs.

use nalgebra::{ComplexField, DVector, RealField};
use num_traits::Float;

use super::{DiscreteSystem, Siso};
use crate::{Discrete, StateSpace, StateSpaceError};

/// Marker for a `DiscreteSystem` with any number of inputs and outputs: `update(&[T]) -> Result<Vec<T>>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mimo;

impl<T: Float + ComplexField + RealField> DiscreteSystem<T, Mimo> {
    /// One sampling period with input `u[k]`: returns (and stores in `output`) `y[k] = C x[k] + D u[k]`,
    /// the output at the same instant as the input, then advances the state to `x[k+1] = A x[k] + B u[k]`.
    pub fn update(&mut self, u: &[T]) -> Result<Vec<T>, StateSpaceError> {
        if u.len() != self.model.order.input {
            return Err(StateSpaceError::InputVector {
                expected_row: self.model.order.input,
                actual_rows: u.len()
            })
        }

        let u: DVector<T> = DVector::from_row_slice(u);
        self.output = (&self.model.c * &self.state) + (&self.model.d * &u);
        self.state = (&self.model.a * &self.state) + (&self.model.b * &u);

        Ok(self.output.as_slice().to_vec())
    }
}

impl<T: Float + ComplexField + RealField> From<StateSpace<T, Discrete>> for DiscreteSystem<T, Mimo> {
    /// Runs `model`, starting at rest.
    fn from(model: StateSpace<T, Discrete>) -> Self {
        Self::at_rest(model)
    }
}

impl<T: Float + ComplexField + RealField> From<&StateSpace<T, Discrete>> for DiscreteSystem<T, Mimo> {
    /// Runs a copy of `model`, starting at rest.
    fn from(model: &StateSpace<T, Discrete>) -> Self {
        Self::at_rest(model.clone())
    }
}

impl<T: Float + ComplexField + RealField> From<DiscreteSystem<T, Siso>> for DiscreteSystem<T, Mimo> {
    /// The same system with its state.
    fn from(system: DiscreteSystem<T, Siso>) -> Self {
        system.with_kind()
    }
}
