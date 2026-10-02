//! Motion state of an axis.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use num_traits::Float;

/// Motion state of an axis. Arithmetic operators act element-wise (`+`, `-`, `*` / `/` by a scalar).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Motion<T> {
    /// Position.
    pub x: T,
    /// Velocity.
    pub v: T,
    /// Acceleration.
    pub a: T,
    /// Force (or torque).
    pub f: T
}

impl<T: Float> Motion<T> {
    /// All zeros.
    pub fn new() -> Self {
        Self { x: T::zero(), v: T::zero(), a: T::zero(), f: T::zero() }
    }
}

impl<T: Float> Add for Motion<T> {
    type Output = Motion<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            v: self.v + rhs.v,
            a: self.a + rhs.a,
            f: self.f + rhs.f,
        }
    }
}

impl<T: Float + AddAssign> AddAssign for Motion<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.v += rhs.v;
        self.a += rhs.a;
        self.f += rhs.f;
    }
}

impl<T: Float> Sub for Motion<T> {
    type Output = Motion<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            v: self.v - rhs.v,
            a: self.a - rhs.a,
            f: self.f - rhs.f,
        }
    }
}

impl<T: Float + SubAssign> SubAssign for Motion<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.v -= rhs.v;
        self.a -= rhs.a;
        self.f -= rhs.f;
    }
}

impl<T: Float> Mul<T> for Motion<T> {
    type Output = Motion<T>;
    fn mul(self, k: T) -> Self::Output {
        Self {
            x: self.x * k,
            v: self.v * k,
            a: self.a * k,
            f: self.f * k,
        }
    }
}

impl<T: Float + MulAssign> MulAssign<T> for Motion<T> {
    fn mul_assign(&mut self, k: T) {
        self.x *= k;
        self.v *= k;
        self.a *= k;
        self.f *= k;
    }
}

impl<T: Float> Div<T> for Motion<T> {
    type Output = Motion<T>;
    fn div(self, k: T) -> Self::Output {
        Self {
            x: self.x / k,
            v: self.v / k,
            a: self.a / k,
            f: self.f / k,
        }
    }
}

impl<T: Float + DivAssign> DivAssign<T> for Motion<T> {
    fn div_assign(&mut self, k: T) {
        self.x /= k;
        self.v /= k;
        self.a /= k;
        self.f /= k;
    }
}
