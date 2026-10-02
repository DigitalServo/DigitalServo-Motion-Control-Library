//! Arithmetic on `TransferFunction`: `G1 * G2`, `G1 / G2`, `G1 + G2`, constant gains (`G * k`, `G / k`, ...)
//! and constant offsets (`G + k`).

use super::TransferFunction;
use crate::Polynomial;
use num_traits::Float;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};

// ---- Transfer function × transfer function -------------------------------------------------

impl<T: Float + AddAssign, D> Mul for &TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn mul(self, rhs: &TransferFunction<T, D>) -> TransferFunction<T, D> {
        TransferFunction::from_polynomials(
            &self.numerator * &rhs.numerator,
            &self.denominator * &rhs.denominator,
        )
        .reduced()
    }
}

impl<T: Float + AddAssign, D> Mul for TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn mul(self, rhs: TransferFunction<T, D>) -> TransferFunction<T, D> {
        &self * &rhs
    }
}

impl<T: Float + AddAssign, D> Div for &TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn div(self, rhs: &TransferFunction<T, D>) -> TransferFunction<T, D> {
        // (n1/d1) / (n2/d2) = (n1/d1) * (d2/n2)
        let rhs_inv = TransferFunction::from_polynomials(rhs.denominator.clone(), rhs.numerator.clone());
        self * &rhs_inv
    }
}

impl<T: Float + AddAssign, D> Div for TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn div(self, rhs: TransferFunction<T, D>) -> TransferFunction<T, D> {
        &self / &rhs
    }
}

impl<T: Float + AddAssign, D> Add for &TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn add(self, rhs: &TransferFunction<T, D>) -> TransferFunction<T, D> {
        // n1/d1 + n2/d2 = (n1*d2 + n2*d1) / (d1*d2); `reduced()` then cancels any factor
        // shared by d1 and d2 (and any other common numerator/denominator roots), which is
        // equivalent to reducing to a common denominator first.
        let n1d2 = &self.numerator * &rhs.denominator;
        let n2d1 = &self.denominator * &rhs.numerator;
        let numerator = &n1d2 + &n2d1;
        let denominator = &self.denominator * &rhs.denominator;
        TransferFunction::from_polynomials(numerator, denominator).reduced()
    }
}

impl<T: Float + AddAssign, D> Add for TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn add(self, rhs: TransferFunction<T, D>) -> TransferFunction<T, D> {
        &self + &rhs
    }
}

// ---- Constant gain -------------------------------------------------------------------------
// Only the numerator is scaled. Poles/zeros are unchanged, so no `reduced()`.
// `*=` / `/=` do the work; `*` / `/` are built on them. (`&G` is copied with `from_polynomials`
// rather than `clone()`, since the derived `Clone` would require `D: Clone`.)

impl<T: Float, D> MulAssign<T> for TransferFunction<T, D> {
    fn mul_assign(&mut self, k: T) {
        self.numerator.iter_mut().for_each(|c| *c = *c * k);
    }
}

impl<T: Float, D> DivAssign<T> for TransferFunction<T, D> {
    fn div_assign(&mut self, k: T) {
        self.numerator.iter_mut().for_each(|c| *c = *c / k);
    }
}

impl<T: Float, D> Mul<T> for TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn mul(mut self, k: T) -> TransferFunction<T, D> {
        self *= k;
        self
    }
}

impl<T: Float, D> Mul<T> for &TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn mul(self, k: T) -> TransferFunction<T, D> {
        TransferFunction::from_polynomials(self.numerator.clone(), self.denominator.clone()) * k
    }
}

impl<T: Float, D> Div<T> for TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn div(mut self, k: T) -> TransferFunction<T, D> {
        self /= k;
        self
    }
}

impl<T: Float, D> Div<T> for &TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn div(self, k: T) -> TransferFunction<T, D> {
        TransferFunction::from_polynomials(self.numerator.clone(), self.denominator.clone()) / k
    }
}

// ---- Feedback ------------------------------------------------------------------------------

impl<T: Float + AddAssign, D> TransferFunction<T, D> {
    /// Closed-loop transfer function of `self` (open loop `L`) under unity negative feedback,
    /// `L / (1 + L)`. Computed directly as `n / (d + n)`; `n + d` shares a root with `n` only if
    /// `d` does, so no `reduced()` is needed when `self` is already reduced.
    pub fn unity_feedback(&self) -> Self {
        TransferFunction::from_polynomials(self.numerator.clone(), &self.denominator + &self.numerator)
    }
}

// ---- Constant offset -----------------------------------------------------------------------
// n/d + k = (n + k*d) / d. Poles are unchanged, and n + k*d shares a root with d only if n does,
// so no `reduced()` is needed.

impl<T: Float + AddAssign, D> AddAssign<T> for TransferFunction<T, D> {
    fn add_assign(&mut self, k: T) {
        let kd = Polynomial(self.denominator.iter().map(|&c| c * k).collect());
        self.numerator += &kd;
    }
}

impl<T: Float + AddAssign, D> Add<T> for TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn add(mut self, k: T) -> TransferFunction<T, D> {
        self += k;
        self
    }
}

impl<T: Float + AddAssign, D> Add<T> for &TransferFunction<T, D> {
    type Output = TransferFunction<T, D>;
    fn add(self, k: T) -> TransferFunction<T, D> {
        TransferFunction::from_polynomials(self.numerator.clone(), self.denominator.clone()) + k
    }
}

// `k * G` / `k + G`: the orphan rule forbids `impl<T> Mul<TransferFunction<T, D>> for T`, so the
// left-hand scalar is implemented per concrete float type. Only f64: with both f32 and f64,
// `10.0 * &g` fails to infer whenever `g`'s `T` is still an unresolved `{float}` (e.g. built by
// `tf!` or from literal slices). For f32, use `g * k` / `g + k`.
macro_rules! impl_scalar_lhs_ops {
    ($($t:ty),*) => {$(
        impl<D> Mul<TransferFunction<$t, D>> for $t {
            type Output = TransferFunction<$t, D>;
            fn mul(self, tf: TransferFunction<$t, D>) -> TransferFunction<$t, D> {
                tf * self
            }
        }

        impl<D> Mul<&TransferFunction<$t, D>> for $t {
            type Output = TransferFunction<$t, D>;
            fn mul(self, tf: &TransferFunction<$t, D>) -> TransferFunction<$t, D> {
                tf * self
            }
        }

        impl<D> Add<TransferFunction<$t, D>> for $t {
            type Output = TransferFunction<$t, D>;
            fn add(self, tf: TransferFunction<$t, D>) -> TransferFunction<$t, D> {
                tf + self
            }
        }

        impl<D> Add<&TransferFunction<$t, D>> for $t {
            type Output = TransferFunction<$t, D>;
            fn add(self, tf: &TransferFunction<$t, D>) -> TransferFunction<$t, D> {
                tf + self
            }
        }
    )*};
}

impl_scalar_lhs_ops!(f64);
