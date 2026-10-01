//! Multirate perfect tracking control (PTC) on the lifted model.
//!
//! The discretized plant `x[k+1] = A x[k] + B u[k]` is lifted over a frame of `n` samples
//! (`n` = number of states): `x[(i+1)n] = A^n x[in] + B_lifted [u[in], ..., u[in+n-1]]`, so the
//! `n` inputs of a frame that bring the state exactly onto a reference are
//! `B_lifted^-1 (x_d[(i+1)n] - A^n x_d[in])`.

use std::borrow::Borrow;

use nalgebra::{ComplexField, DMatrix, DVector, RealField};
use num_traits::Float;
use thiserror::Error;

use super::ReferenceSignal;
use crate::discretize::exact_discretize::DiscretizedSystem;
use crate::laplace_transform::StableInverseError;
use crate::{Continuous, Discrete, StateSpace, StateSpaceError};

pub struct LiftedDiscretizedSystem<T> {
    /// Continuous-time model the system was discretized from.
    pub continuous: StateSpace<T, Continuous>,
    pub ssr: StateSpace<T, Discrete>,
    pub ts: T,
    pub order: u32,
    inv_b: DMatrix<T>,
}

impl<T: Float + ComplexField + RealField> TryInto<LiftedDiscretizedSystem<T>> for DiscretizedSystem<T> {
    type Error = StateSpaceError;

    fn try_into(self) -> Result<LiftedDiscretizedSystem<T>, Self::Error> {
        let system = self.borrow();

        let n = system.ssr.order.system;
        let m = system.ssr.order.input;

        let order = n as u32;

        let a = &system.ssr.a;
        let b = &system.ssr.b;

        let a_lifted = a.pow(order);

        // b_lifted = [a^(order-1)*b, a^(order-2)*b, ..., a*b, b]
        let mut b_lifted = DMatrix::<T>::zeros(n, m * order as usize);
        let mut a_power = DMatrix::<T>::identity(n, n);
        for k in 0..order as usize {
            let col = (order as usize - 1 - k) * m;
            b_lifted.columns_mut(col, m).copy_from(&(&a_power * b));
            a_power = a * &a_power;
        }

        let d = DMatrix::<T>::zeros(system.ssr.order.output, order as usize);

        let inv_b = b_lifted.clone()
            .try_inverse()
            .ok_or(StateSpaceError::SingularMatrix)?;

        let ssr = StateSpace::new(a_lifted.clone(), b_lifted.clone(), system.ssr.c.clone(), d)?;

        Ok(LiftedDiscretizedSystem {
            continuous: system.continuous.clone(),
            ssr,
            ts: system.ts,
            order,
            inv_b
        })
    }
}


#[derive(Clone, Debug, Error, PartialEq)]
pub enum PtcError {
    #[error("Perfect tracking from an output reference supports SISO systems only, got {inputs} inputs and {outputs} outputs")]
    NotSiso { inputs: usize, outputs: usize },

    #[error("The system has a direct feedthrough term (D != 0)")]
    Feedthrough,

    #[error("(A, B) is not controllable")]
    Uncontrollable,

    #[error(transparent)]
    StableInverse(#[from] StableInverseError),
}

impl<T: Float + ComplexField + RealField> LiftedDiscretizedSystem<T> {
    /// Perfect tracking control input from a state reference given at every sample:
    /// `u[k] = B_lifted^-1 (r[(k+1)n] - A^n r[kn])` for each frame of `n` samples
    /// (only every `n`-th element of `r` is used).
    pub fn calculate_ptc_input_for_reference_state(&self, r: Vec<Vec<T>>) -> Vec<T> {
        let frames: Vec<DVector<T>> = r
            .chunks(self.order as usize)
            .map(|x| DVector::from(x[0].clone()))
            .collect();
        self.inputs_from_frame_states(&frames)
    }

    /// `B_lifted^-1 (x[k+1] - A^n x[k])` for consecutive frame states.
    fn inputs_from_frame_states(&self, frames: &[DVector<T>]) -> Vec<T> {
        let mut u = Vec::<T>::with_capacity(frames.len() * self.order as usize);
        for points in frames.windows(2) {
            let u_effort = &points[1] - &self.ssr.a * &points[0];
            let u_local = &self.inv_b * u_effort;
            u.extend(u_local.iter());
        }
        u
    }

    /// Perfect tracking control input that makes the output follow `y_d`, for `samples` samples
    /// (rounded up to whole frames of `n` samples). `u[k]` is applied at `t = k ts`, on the same
    /// time axis as `y_d`: place the move on it with the trajectory's start time (rest time).
    ///
    /// The state reference is obtained by stable inversion (bilateral Laplace transform) of the
    /// continuous-time plant `G(s) = C (sI - A)^-1 B`, see `TransferFunction::state_reference_from_output`.
    /// Unstable zeros of `G` make it non-causal: the input starts before the trajectory does
    /// (pre-actuation), so leave enough rest time before the move.
    pub fn calculate_ptc_input_for_reference_output<S: ReferenceSignal<T>>(
        &self,
        y_d: &S,
        samples: usize,
    ) -> Result<Vec<T>, PtcError> {
        let reference = self.state_reference_from_output(y_d)?;
        let n = self.order as usize;
        let frame_time = self.ts * T::from(n).unwrap();
        let frames: Vec<DVector<T>> = (0..=samples.div_ceil(n))
            .map(|k| reference(frame_time * T::from(k).unwrap()))
            .collect();
        Ok(self.inputs_from_frame_states(&frames))
    }

    /// State reference `x_d(t)` (in this system's state coordinates) that makes the output follow `y_d`.
    pub fn state_reference_from_output<S: ReferenceSignal<T>>(&self, y_d: &S) -> Result<impl Fn(T) -> DVector<T> + use<T, S>, PtcError> {
        let sys = &self.continuous;
        if sys.order.input != 1 || sys.order.output != 1 {
            return Err(PtcError::NotSiso { inputs: sys.order.input, outputs: sys.order.output });
        }
        if sys.d.iter().any(|d| !d.is_zero()) {
            return Err(PtcError::Feedthrough);
        }

        let plant = sys.transfer_function().map_err(|_| PtcError::NotSiso { inputs: sys.order.input, outputs: sys.order.output })?;
        let reference = plant.state_reference_from_output(y_d)?;

        // x = T x_c with the controllable canonical realization (A_c, B_c) (in which `StateReference`
        // is expressed): T = W(A, B) W(A_c, B_c)^-1.
        let canonical = StateSpace::normalized_controllable_canonical(&plant).map_err(|_| PtcError::Uncontrollable)?;
        let w = controllability_matrix(&sys.a, &sys.b);
        let w_c = controllability_matrix(&canonical.a, &canonical.b);
        let w_c_inv = w_c.try_inverse().ok_or(PtcError::Uncontrollable)?;
        let transform = w * w_c_inv;
        if transform.clone().try_inverse().is_none() {
            return Err(PtcError::Uncontrollable);
        }

        Ok(move |t: T| &transform * DVector::from(reference.state(t)))
    }
}

/// `[B, AB, ..., A^(n-1) B]` (single input).
fn controllability_matrix<T: Float + ComplexField + RealField>(a: &DMatrix<T>, b: &DMatrix<T>) -> DMatrix<T> {
    let n = a.nrows();
    let mut w = DMatrix::<T>::zeros(n, n);
    let mut column = b.column(0).into_owned();
    for k in 0..n {
        w.set_column(k, &column);
        column = a * column;
    }
    w
}
