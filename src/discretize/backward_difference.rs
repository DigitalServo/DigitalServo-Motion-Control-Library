//! Backward difference (backward Euler) `s = (z - 1)/(ts z)`.

use std::convert::Infallible;
use std::ops::{AddAssign, MulAssign};
use num_traits::Float;

use nalgebra::{ComplexField, DMatrix, RealField};

use super::Method;
use crate::{Continuous, Discrete, Polynomial, StateSpace, StateSpaceError, TransferFunction};
use crate::math::binomial_coefficient;

/// Backward difference (backward Euler) `s = (z - 1)/(ts z) = (1 - z^-1)/ts`.
///
/// - `TransferFunction`: substitution into a proper `G(s)`; the result is normalized so that the
///   leading denominator coefficient is 1. It cannot fail (`Error = Infallible`).
/// - `StateSpace`: with `M = (I - A ts)^-1`, `A_d = M`, `B_d = M B ts`, `C_d = C M`,
///   `D_d = D + C M B ts`, so that `C_d (zI - A_d)^-1 B_d + D_d = G((z - 1)/(ts z))`
///   (any number of inputs and outputs). `SingularMatrix` if `A` has the eigenvalue `1/ts`.
///
/// A pole `p` maps to `1/(1 - p ts)`: the stable region maps into the disk `|z - 1/2| < 1/2`, so a
/// stable system stays stable, and the DC gain is kept. Unlike `Tustin`, the result always has a
/// direct term (it is biproper) and the frequency response is not exact even on a warped axis.
///
/// ```
/// use dsmc::{tf, DiscreteSystem, discretize::BackwardDifference};
///
/// let g_z = tf!("100 / (s + 100)").discretize(BackwardDifference, 1e-3).unwrap();
/// let mut filter = DiscreteSystem::try_from(&g_z).unwrap();
/// let y = filter.update(1.0);
/// ```
///
/// Numerator: bm*s^m + bm-1*s^m-1 + ...+ b0 => \[bm, bm-1, ..., b0\]
///
/// Denominator: an*s^n + an-1*s^n-1 + ...+ a0 => \[an, an-1, ..., a0\]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackwardDifference;

impl<T: Float + AddAssign + MulAssign> Method<T, TransferFunction<T, Continuous>> for BackwardDifference {
    type Output = TransferFunction<T, Discrete>;
    type Error = Infallible;

    fn apply(&self, tf: &TransferFunction<T, Continuous>, ts: T) -> Result<Self::Output, Self::Error> {
        Ok(transform(tf, ts))
    }
}

impl<T: Float + ComplexField + RealField> Method<T, StateSpace<T, Continuous>> for BackwardDifference {
    type Output = StateSpace<T, Discrete>;
    type Error = StateSpaceError;

    fn apply(&self, system: &StateSpace<T, Continuous>, ts: T) -> Result<Self::Output, Self::Error> {
        let n = system.order.system;
        let identity = DMatrix::<T>::identity(n, n);
        let m = if n == 0 { identity.clone() } else { (&identity - system.a.scale(ts)).try_inverse().ok_or(StateSpaceError::SingularMatrix)? };

        let b = (&m * &system.b).scale(ts);
        let c = &system.c * &m;
        let d = &system.d + (&c * &system.b).scale(ts);
        StateSpace::new(m, b, c, d)
    }
}

/// Coefficients of `sum_k ck α^k (z - 1)^k z^(n - k)` (`coeffs` in descending powers of `s`),
/// i.e. `(ts z)^n` times the polynomial in `s = α (z - 1)/z`, `α = 1/ts`.
fn substitute<T: Float + AddAssign>(coeffs: &Polynomial<T>, n: usize, alpha: T) -> Polynomial<T> {
    let mut poly = Polynomial::zeros(n);
    for (k, &ck) in coeffs.iter().rev().enumerate() {
        if ck != T::zero() {
            let scaler = ck * alpha.powi(k as i32);
            // (z - 1)^k z^(n - k) = sum_i C(k, i) (-1)^i z^(n - i)
            for i in 0..=k {
                let term = scaler * T::from(binomial_coefficient(k, i)).unwrap();
                poly[i] += if i % 2 == 0 { term } else { -term };
            }
        }
    }
    poly
}

fn transform<T: Float + AddAssign + MulAssign>(tf: &TransferFunction<T, Continuous>, ts: T) -> TransferFunction<T, Discrete> {
    let n = tf.denominator.len() - 1;
    let alpha = T::one() / ts;

    let mut numer_z = substitute(&tf.numerator, n, alpha);
    let mut denom_z = substitute(&tf.denominator, n, alpha);

    let scale = T::one() / denom_z[0];
    numer_z *= scale;
    denom_z *= scale;

    TransferFunction::from_polynomials(numer_z, denom_z)
}
