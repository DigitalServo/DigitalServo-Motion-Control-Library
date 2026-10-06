//! A discrete-time system running sample by sample (plant model, controller, filter).
//!
//! The struct and what both kinds share are here; each kind (`Mimo`, `Siso`) has its own file
//! with its marker, its `update` and the conversions that build it.

mod mimo;
mod siso;

pub use mimo::Mimo;
pub use siso::Siso;

use std::marker::PhantomData;

use nalgebra::{ComplexField, DVector, RealField};
use num_traits::Float;

use crate::{Discrete, StateSpace};

/// A discrete-time system running one sample at a time, e.g. a simulated plant or the
/// implementation of a digital controller or filter: a copy of the model with its state, so that
/// `StateSpace` and `TransferFunction` stay plain representations. It starts at rest.
///
/// `K` is `Mimo` (the default; any number of inputs and outputs) or `Siso` (single input and
/// output, checked once when it is built, so that `update` takes and returns a scalar and does
/// not allocate):
///
/// - from a `TransferFunction<T, Discrete>` (`TryFrom`): `Siso`, through its controllable
///   canonical realization (the difference equation in state form); only an improper transfer
///   function is an error;
/// - from a `StateSpace<T, Discrete>`: `Mimo` with `From`, `Siso` with `TryFrom` (`NotSiso` otherwise);
/// - between the two (`From` / `TryFrom`), keeping the state.
///
/// ```
/// use dsmc::{tf, DiscreteSystem, StateSpace, discretize::Zoh};
///
/// let g = tf!("100 / (s + 100)");
/// let mut plant = DiscreteSystem::try_from(&g.discretize(Zoh, 1e-3).unwrap()).unwrap();
/// let y: Vec<f64> = (0..100).map(|_| plant.update(1.0)).collect();
///
/// let ss = StateSpace::try_from(&g).unwrap().discretize(Zoh, 1e-3).unwrap();
/// let mut plant = DiscreteSystem::from(ss);
/// let y = plant.update(&[1.0]).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct DiscreteSystem<T, K = Mimo> {
    model: StateSpace<T, Discrete>,
    /// Current state `x[k]`.
    pub state: DVector<T>,
    /// Output `y[k]` of the last `update`.
    pub output: DVector<T>,
    /// Work space for `x[k+1]` (`Siso`).
    next_state: DVector<T>,
    _kind: PhantomData<K>,
}

impl<T: Float + ComplexField + RealField, K> DiscreteSystem<T, K> {
    /// The model (state-space representation) being run.
    pub fn model(&self) -> &StateSpace<T, Discrete> {
        &self.model
    }

    /// Back to rest: `x = 0`, `y = 0`.
    pub fn reset(&mut self) {
        self.state.fill(T::zero());
        self.output.fill(T::zero());
    }

    /// Runs `model`, starting at rest (the dimensions are not checked against `K`).
    fn at_rest(model: StateSpace<T, Discrete>) -> Self {
        let state = DVector::zeros(model.order.system);
        let output = DVector::zeros(model.order.output);
        let next_state = state.clone();
        Self { model, state, output, next_state, _kind: PhantomData }
    }

    /// The same system with its state, as kind `L` (the dimensions are not checked against `L`).
    fn with_kind<L>(self) -> DiscreteSystem<T, L> {
        let Self { model, state, output, next_state, .. } = self;
        DiscreteSystem { model, state, output, next_state, _kind: PhantomData }
    }
}
