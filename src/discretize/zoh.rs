//! Zero-order hold (step-invariant, exact) discretization.

use std::borrow::Borrow;

use nalgebra::{ComplexField, DMatrix, DVector, RealField};
use num_traits::Float;

use crate::{Continuous, Discrete, StateSpace, StateSpaceError, TransferFunction};

/// Zero-order hold discretization `A_d = e^(A ts)`, `B_d = ∫_0^ts e^(A τ) dτ B` (computed as one matrix
/// exponential of the augmented matrix `[[A, B], [0, 0]] ts`); `C` and `D` are unchanged.
pub fn discretize_ssr<T: Float + ComplexField + RealField, S: Borrow<StateSpace<T, Continuous>>>(ssr: S, ts: T) -> Result<StateSpace<T, Discrete>, StateSpaceError> {
    let system = ssr.borrow();

    // Augmented matrix method
    let (a, b) = {
        let n = system.order.system;
        let m = system.order.input;

        let mut aug = DMatrix::<T>::zeros(n + m, n + m);
        aug.view_mut((0, 0), (n, n)).copy_from(&system.a);
        aug.view_mut((0, n), (n, m)).copy_from(&system.b);

        let aug_exp = aug.scale(ts).exp();

        let a = aug_exp.view((0, 0), (n, n)).into_owned();
        let b = aug_exp.view((0, n), (n, m)).into_owned();

        (a, b)
    };

    let c = system.c.clone();
    let d = system.d.clone();

    StateSpace::new(a, b, c, d)
}

/// Step-invariant (zero-order hold) discretization of a transfer function:
/// `G(z) = (1 - z^-1) Z[L^-1[G(s) / s]]`, computed through the state space
/// (controllable canonical realization, `discretize_ssr`, then `C (zI - A_d)^-1 B_d + D`).
/// A static gain is returned as is.
///
/// The result is exact at the sampling instants when the input is held constant between samples
/// (the usual situation of a digital controller driving a plant through a D/A converter).
///
/// ```
/// use dsmc::{tf, discretize::zoh};
///
/// let g_z = zoh::discretize(&tf!("100 / (s + 100)"), 1e-3).unwrap();
/// ```
pub fn discretize<T, S>(tf: S, ts: T) -> Result<TransferFunction<T, Discrete>, StateSpaceError>
where
    T: Float + ComplexField + RealField,
    S: Borrow<TransferFunction<T, Continuous>>,
{
    let tf = tf.borrow();
    match StateSpace::controllable_canonical(tf) {
        Ok(ssr) => discretize_ssr(&ssr, ts)?.transfer_function(),
        Err(StateSpaceError::EmptySystem) => {
            Ok(TransferFunction::from_polynomials(tf.numerator.clone(), tf.denominator.clone()))
        }
        Err(e) => Err(e),
    }
}

/// Zero-order-hold discretization of a continuous-time state-space model, with its state,
/// for sample-by-sample simulation (`update`).
#[derive(Clone)]
pub struct DiscretizedSystem<T> {
    /// Continuous-time model the system was discretized from.
    pub continuous: StateSpace<T, Continuous>,
    /// Discretized model (`discretize_ssr`).
    pub ssr: StateSpace<T, Discrete>,
    /// Current state `x[k]`.
    pub state: DVector<T>,
    /// Output `y[k]` of the last `update`.
    pub output: DVector<T>,
    /// Sampling period.
    pub ts: T,
}

impl<T: Float + ComplexField + RealField> DiscretizedSystem<T> {
    /// Discretize `ssr` with sampling period `ts` (`discretize_ssr`), starting at rest.
    pub fn from_ssr<S: Borrow<StateSpace<T, Continuous>>>(ssr: S, ts: T) -> Result<Self, StateSpaceError> {
        let continuous = ssr.borrow().clone();
        let ssr = discretize_ssr(&continuous, ts)?;
        let state = DVector::zeros(ssr.order.system);
        let output = DVector::zeros(ssr.order.output);

        Ok(Self { continuous, ssr, state, output, ts})
    }

    /// Discretized controllable canonical realization of `tf_c` (monic denominator,
    /// `B = [0, ..., 0, 1]^T`; see `StateSpace::controllable_canonical`). Works for any proper `tf_c`.
    pub fn from_tf<S: Borrow<TransferFunction<T, Continuous>>>(tf_c: S, ts: T) -> Result<Self, StateSpaceError> {
        Self::from_ssr(StateSpace::controllable_canonical(tf_c.borrow())?, ts)
    }

    /// Discretized controllable canonical realization of `tf_c` whose output equals a state at low
    /// frequency (`x = u / D(s)`, `y = N(s) x` with `N` normalized; `y = x_1` at DC if `N(0) != 0`,
    /// and exactly without zeros; see `StateSpace::normalized_controllable_canonical`). These are the
    /// state coordinates of `ReferenceSignal::to_state_reference`.
    pub fn from_tf_normalized<S: Borrow<TransferFunction<T, Continuous>>>(tf_c: S, ts: T) -> Result<Self, StateSpaceError> {
        Self::from_ssr(StateSpace::normalized_controllable_canonical(tf_c.borrow())?, ts)
    }

    /// One sampling period with input `u[k]`: returns (and stores in `output`) `y[k] = C x[k] + D u[k]`,
    /// the output at the same instant as the input, then advances the state to `x[k+1] = A x[k] + B u[k]`.
    pub fn update(&mut self, u: &[T]) -> Result<Vec<T>, StateSpaceError> {
        if u.len() != self.ssr.order.input {
            return Err(StateSpaceError::InputVector {
                expected_row: self.ssr.order.input,
                actual_rows: u.len()
            })
        }

        let u: DVector<T> = DVector::from_row_slice(u);
        self.output = (&self.ssr.c * &self.state) + (&self.ssr.d * &u);
        self.state = (&self.ssr.a * &self.state) + (&self.ssr.b * &u);

        Ok(self.output.as_slice().to_vec())
    }
}
