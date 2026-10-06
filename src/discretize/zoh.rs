//! Zero-order hold (step-invariant, exact) discretization.

use nalgebra::{ComplexField, DMatrix, RealField};
use num_traits::Float;

use super::Method;
use crate::{Continuous, Discrete, StateSpace, StateSpaceError, TransferFunction};

/// Zero-order hold (step-invariant) discretization of a state-space model or a transfer function,
/// exact at the sampling instants when the input is held constant between samples (the usual
/// situation of a digital controller driving a plant through a D/A converter).
///
/// - `StateSpace`: `A_d = e^(A ts)`, `B_d = ∫_0^ts e^(A τ) dτ B` (computed as one matrix exponential
///   of the augmented matrix `[[A, B], [0, 0]] ts`); `C` and `D` are unchanged.
/// - `TransferFunction`: `G(z) = (1 - z^-1) Z[L^-1[G(s) / s]]`, computed through the state space
///   (controllable canonical realization, then `C (zI - A_d)^-1 B_d + D`). A static gain is
///   returned as is.
///
/// The result is run sample by sample with `DiscreteSystem`.
///
/// ```
/// use dsmc::{tf, DiscreteSystem, StateSpace, discretize::Zoh};
///
/// let g = tf!("100 / (s + 100)");
/// let g_z = g.discretize(Zoh, 1e-3).unwrap();
///
/// let mut plant = DiscreteSystem::from(StateSpace::try_from(&g).unwrap().discretize(Zoh, 1e-3).unwrap());
/// let y = plant.update(&[1.0]).unwrap();
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Zoh;

impl<T: Float + ComplexField + RealField> Method<T, StateSpace<T, Continuous>> for Zoh {
    type Output = StateSpace<T, Discrete>;
    type Error = StateSpaceError;

    fn apply(&self, system: &StateSpace<T, Continuous>, ts: T) -> Result<Self::Output, Self::Error> {
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

        StateSpace::new(a, b, system.c.clone(), system.d.clone())
    }
}

impl<T: Float + ComplexField + RealField> Method<T, TransferFunction<T, Continuous>> for Zoh {
    type Output = TransferFunction<T, Discrete>;
    type Error = StateSpaceError;

    fn apply(&self, tf: &TransferFunction<T, Continuous>, ts: T) -> Result<Self::Output, Self::Error> {
        match StateSpace::controllable_canonical(tf) {
            Ok(ssr) => self.apply(&ssr, ts)?.transfer_function(),
            Err(StateSpaceError::EmptySystem) => {
                Ok(TransferFunction::from_polynomials(tf.numerator.clone(), tf.denominator.clone()))
            }
            Err(e) => Err(e),
        }
    }
}
