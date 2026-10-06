//! `DiscreteSystem<T, Siso>`: single input and output, scalar `update` without allocation.

use nalgebra::{ComplexField, RealField};
use num_traits::Float;

use super::{DiscreteSystem, Mimo};
use crate::{Discrete, StateSpace, StateSpaceError, TransferFunction};

/// Marker for a single-input single-output `DiscreteSystem`: `update(T) -> T`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Siso;

impl<T: Float + ComplexField + RealField> DiscreteSystem<T, Siso> {
    /// One sampling period with input `u[k]`: returns (and stores in `output[0]`)
    /// `y[k] = c x[k] + d u[k]`, the output at the same instant as the input, then advances the
    /// state to `x[k+1] = A x[k] + b u[k]` (without allocation).
    pub fn update(&mut self, u: T) -> T {
        let model = &self.model;
        let y = model.c.row(0).iter().zip(self.state.iter()).fold(model.d[(0, 0)] * u, |acc, (&c, &x)| acc + c * x);
        self.next_state.gemv(T::one(), &model.a, &self.state, T::zero());
        self.next_state.axpy(u, &model.b.column(0), T::one());
        std::mem::swap(&mut self.state, &mut self.next_state);
        self.output[0] = y;
        y
    }
}

impl<T: Float + ComplexField + RealField> TryFrom<StateSpace<T, Discrete>> for DiscreteSystem<T, Siso> {
    type Error = StateSpaceError;

    /// Runs `model`, starting at rest; `NotSiso` unless it has one input and one output.
    fn try_from(model: StateSpace<T, Discrete>) -> Result<Self, Self::Error> {
        DiscreteSystem::<T, Mimo>::at_rest(model).try_into()
    }
}

impl<T: Float + ComplexField + RealField> TryFrom<&StateSpace<T, Discrete>> for DiscreteSystem<T, Siso> {
    type Error = StateSpaceError;

    /// Runs a copy of `model`, starting at rest; `NotSiso` unless it has one input and one output.
    fn try_from(model: &StateSpace<T, Discrete>) -> Result<Self, Self::Error> {
        Self::try_from(model.clone())
    }
}

impl<T: Float + ComplexField + RealField> TryFrom<&TransferFunction<T, Discrete>> for DiscreteSystem<T, Siso> {
    type Error = StateSpaceError;

    /// Runs the controllable canonical realization of `tf` (`StateSpace::try_from`; a static gain
    /// has no state), starting at rest. An improper `tf` (not causal) is an error.
    fn try_from(tf: &TransferFunction<T, Discrete>) -> Result<Self, Self::Error> {
        Self::try_from(StateSpace::try_from(tf)?)
    }
}

impl<T: Float + ComplexField + RealField> TryFrom<DiscreteSystem<T, Mimo>> for DiscreteSystem<T, Siso> {
    type Error = StateSpaceError;

    /// The same system with its state; `NotSiso` unless it has one input and one output.
    fn try_from(system: DiscreteSystem<T, Mimo>) -> Result<Self, Self::Error> {
        let order = &system.model.order;
        if order.input != 1 || order.output != 1 {
            return Err(StateSpaceError::NotSiso { inputs: order.input, outputs: order.output });
        }
        Ok(system.with_kind())
    }
}
