use std::ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign};
use num_traits::Float;

mod dka_method;
pub use dka_method::dka_method;

mod vieta_formula;
pub use vieta_formula::vieta_formula;

/// Coefficients of a decending polynomial
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial<T>(pub Vec<T>);

impl<T: Float> Polynomial<T> {
    pub fn new() -> Self {
        Polynomial(vec![])
    }

    pub fn zeros(degree: usize) -> Self {
        Polynomial(vec![T::zero(); degree + 1])
    }
}

impl<T: Float> Deref for Polynomial<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Float> DerefMut for Polynomial<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Float + AddAssign> Add for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn add(self, rhs: &Polynomial<T>) -> Polynomial<T> {
        let max_len = self.len().max(rhs.len());
        let mut result = vec![T::zero(); max_len];

        // Coefficients are descending-order (index 0 = highest degree), so operands of
        // different lengths must be aligned by degree from the constant-term end.
        for (slot, &v) in result.iter_mut().rev().zip(self.iter().rev()) {
            *slot += v;
        }
        for (slot, &v) in result.iter_mut().rev().zip(rhs.iter().rev()) {
            *slot += v;
        }
        Polynomial(result)
    }
}

impl<T: Float + AddAssign> AddAssign<&Polynomial<T>> for Polynomial<T> {
    fn add_assign(&mut self, rhs: &Polynomial<T>) {
        let max_len = self.len().max(rhs.len());
        if self.len() < max_len {
            // Prepend (not append) zeros, since index 0 is the highest degree.
            self.0.splice(0..0, std::iter::repeat(T::zero()).take(max_len - self.len()));
        }
        let offset = self.len() - rhs.len();
        for (i, &v) in rhs.iter().enumerate() {
            self[offset + i] += v;
        }
    }
}

impl<T: Float + AddAssign> Mul for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn mul(self, rhs: &Polynomial<T>) -> Polynomial<T> {
        if self.is_empty() || rhs.is_empty() {
            return Polynomial(vec![T::zero()]);
        }
        let mut result = vec![T::zero(); self.len() + rhs.len() - 1];
        for (i, &ai) in self.iter().enumerate() {
            for (j, &bj) in rhs.iter().enumerate() {
                result[i + j] += ai * bj;
            }
        }
        Polynomial(result)
    }
}

impl<T: Float + MulAssign> Mul<T> for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn mul(self, rhs: T) -> Polynomial<T> {
        Polynomial(self.iter().map(|&x| x * rhs).collect())
    }
}

impl<T: Float + MulAssign> Mul<T> for Polynomial<T> {
    type Output = Polynomial<T>;
    fn mul(self, rhs: T) -> Polynomial<T> {
        &self * rhs
    }
}

impl<T: Float + MulAssign> MulAssign<T> for Polynomial<T> {
    fn mul_assign(&mut self, rhs: T) {
        for x in self.iter_mut() {
            *x *= rhs;
        }
    }
}

impl<T: Float + AddAssign> MulAssign<&Polynomial<T>> for Polynomial<T> {
    fn mul_assign(&mut self, rhs: &Polynomial<T>) {
        let result = &*self * rhs;
        self.0 = result.0;
    }
}
