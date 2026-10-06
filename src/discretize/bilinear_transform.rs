//! Bilinear (Tustin) transform `s = 2/ts (z - 1)/(z + 1)`.

use std::convert::Infallible;
use std::ops::{AddAssign, MulAssign};
use num_traits::Float;

use nalgebra::{ComplexField, DMatrix, RealField};

use super::Method;
use crate::{Continuous, Discrete, Polynomial, StateSpace, StateSpaceError, TransferFunction};
use crate::math::binomial_coefficient;

/// Descending order of powers for (1 + x)^n
fn binom_one_plus_x<T: Float>(n: usize) -> Polynomial<T> {
    Polynomial((0..=n).map(|k| T::from(binomial_coefficient(n, k)).unwrap()).collect())
}

/// Descending order of powers for (1 - x)^n
fn binom_one_minus_x<T: Float>(n: usize) -> Polynomial<T> {
    let mut poly: Polynomial<T> = binom_one_plus_x(n);
    for i in (1..=n).step_by(2) {
        poly[i] = -poly[i];
    }
    poly
}

/// Bilinear (Tustin) transform `s = 2/ts (z - 1)/(z + 1)`.
///
/// - `TransferFunction`: substitution into a proper `G(s)`; the result is normalized so that the
///   leading denominator coefficient is 1. It cannot fail (`Error = Infallible`).
/// - `StateSpace`: with `M = (I - A ts/2)^-1`, `A_d = M (I + A ts/2)`, `B_d = M B ts`, `C_d = C M`,
///   `D_d = D + C M B ts/2`, so that `C_d (zI - A_d)^-1 B_d + D_d = G(2/ts (z - 1)/(z + 1))`
///   (any number of inputs and outputs). `SingularMatrix` if `A` has the eigenvalue `2/ts`.
///
/// The stable region maps onto the stable region and the gain is kept, while the frequency axis is
/// warped (`ω_d = 2/ts atan(ω ts / 2)`).
///
/// ```
/// use dsmc::{tf, DiscreteSystem, discretize::Tustin};
///
/// let g_z = tf!("100 / (s + 100)").discretize(Tustin, 1e-3).unwrap();
/// let mut filter = DiscreteSystem::try_from(&g_z).unwrap();
/// let y = filter.update(1.0);
/// ```
///
/// Numerator: bm*s^m + bm-1*s^m-1 + ...+ b0 => \[bm, bm-1, ..., b0\]
///
/// Denominator: an*s^n + an-1*s^n-1 + ...+ a0 => \[an, an-1, ..., a0\]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tustin;

impl<T: Float + AddAssign + MulAssign> Method<T, TransferFunction<T, Continuous>> for Tustin {
    type Output = TransferFunction<T, Discrete>;
    type Error = Infallible;

    fn apply(&self, tf: &TransferFunction<T, Continuous>, ts: T) -> Result<Self::Output, Self::Error> {
        Ok(transform(tf, ts))
    }
}

impl<T: Float + ComplexField + RealField> Method<T, StateSpace<T, Continuous>> for Tustin {
    type Output = StateSpace<T, Discrete>;
    type Error = StateSpaceError;

    fn apply(&self, system: &StateSpace<T, Continuous>, ts: T) -> Result<Self::Output, Self::Error> {
        let n = system.order.system;
        let half = ts / T::from(2.0).unwrap();
        let identity = DMatrix::<T>::identity(n, n);
        let a_half = system.a.scale(half);
        let m = if n == 0 { identity.clone() } else { (&identity - &a_half).try_inverse().ok_or(StateSpaceError::SingularMatrix)? };

        let a = &m * (&identity + &a_half);
        let b = (&m * &system.b).scale(ts);
        let c = &system.c * &m;
        let d = &system.d + (&c * &system.b).scale(half);
        StateSpace::new(a, b, c, d)
    }
}

fn transform<T: Float + AddAssign + MulAssign>(tf: &TransferFunction<T, Continuous>, ts: T) -> TransferFunction<T, Discrete> {

    let n = tf.denominator.len() - 1;
    let mut numer_z = Polynomial::zeros(n);
    let mut denom_z = Polynomial::zeros(n);

    let alpha = T::from(2.0).unwrap() / ts;

    // Numerator： sum of [(bk * α^k * (1 - q)^k) * (1 + q)^{N - k}]
    for (k, &bk) in tf.numerator.iter().rev().enumerate() {
        if bk != T::zero() {
            let scaler = bk * alpha.powi(k as i32);
            let term = &binom_one_minus_x(k) * &binom_one_plus_x(n - k);
            numer_z += &(term * scaler);
        }
    }

    // Denominator： sum of [(ak * α^k * (1 - q)^k) * (1 + q)^{N - k}]
    for (k, &ak) in tf.denominator.iter().rev().enumerate() {
        if ak != T::zero() {
            let scalar = ak * alpha.powi(k as i32);
            let term = &binom_one_minus_x(k) * &binom_one_plus_x(n - k);
            denom_z += &(&term * scalar);
        }
    }

    let scale = T::one() / denom_z[0];
    numer_z *= scale;
    denom_z *= scale;

    TransferFunction::from_polynomials(numer_z, denom_z)
}
