use std::ops::{Add, AddAssign, Sub, SubAssign};
use num_traits::Float;

#[derive(Copy, Clone)]
pub struct Motion<T> {
    pub x: T,
    pub v: T,
    pub a: T,
    pub f: T
}

impl<T: Float> Motion<T> {
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
