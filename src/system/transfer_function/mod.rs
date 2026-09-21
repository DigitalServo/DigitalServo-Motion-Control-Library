use crate::{dka_method, vieta_formula, Polynomial};
use num_complex::Complex;
use num_traits::Float;
use std::ops::{Add, AddAssign, Mul};

#[derive(Clone, Debug)]
pub struct TransferFunction<T> {
    pub numerator: Polynomial<T>,
    pub denominator: Polynomial<T>,
}

impl<T: Clone> TransferFunction<T> {
    pub fn new(numerator: &[T], denominator: &[T]) -> Self {
        Self {
            numerator: Polynomial(numerator.to_vec()),
            denominator: Polynomial(denominator.to_vec()),
        }
    }
}

impl<T: Float + AddAssign> TransferFunction<T> {
    /// Cancel poles/zeros shared by the numerator and denominator (pole-zero cancellation).
    /// Roots are found numerically, so a relative tolerance (1e-6) is used to decide whether
    /// a numerator root and a denominator root are "the same" root.
    pub fn reduced(&self) -> Self {
        self.reduced_with_tolerance(T::from(1e-6).unwrap())
    }

    /// Same as `reduced`, but with an explicit relative tolerance for matching roots.
    pub fn reduced_with_tolerance(&self, rel_tol: T) -> Self {
        let (numer_gain, mut numer_roots) = roots_and_gain(&self.numerator);
        let (denom_gain, mut denom_roots) = roots_and_gain(&self.denominator);
        cancel_common_roots(&mut numer_roots, &mut denom_roots, rel_tol);
        Self {
            numerator: reconstruct(numer_gain, &numer_roots),
            denominator: reconstruct(denom_gain, &denom_roots),
        }
    }
}

/// Leading coefficient and roots of a descending-order real polynomial (roots found via
/// a Complex-coefficient embedding, since num-traits::Float has no general polynomial GCD).
fn roots_and_gain<T: Float + AddAssign>(p: &Polynomial<T>) -> (T, Vec<Complex<T>>) {
    let coeffs: Vec<T> = p.0.iter().copied().skip_while(|c| c.is_zero()).collect();
    let Some(&gain) = coeffs.first() else {
        return (T::zero(), Vec::new());
    };
    let complex_poly = Polynomial(coeffs.iter().map(|&c| Complex::new(c, T::zero())).collect());
    let roots = dka_method(&complex_poly).unwrap_or_default();
    (gain, roots)
}

/// Remove matching root pairs (within `rel_tol`, relative to root magnitude) from both lists.
fn cancel_common_roots<T: Float>(a: &mut Vec<Complex<T>>, b: &mut Vec<Complex<T>>, rel_tol: T) {
    let mut i = 0;
    while i < a.len() {
        let threshold = rel_tol * a[i].norm().max(T::one());
        match b.iter().position(|&r| (r - a[i]).norm() <= threshold) {
            Some(j) => {
                a.remove(i);
                b.remove(j);
            }
            None => i += 1,
        }
    }
}

/// Rebuild a real, descending-order polynomial from a leading coefficient and its roots.
fn reconstruct<T: Float>(gain: T, roots: &[Complex<T>]) -> Polynomial<T> {
    let monic = vieta_formula(roots);
    Polynomial(monic.0.iter().map(|c| c.re * gain).collect())
}

impl<T: Float + AddAssign> Mul for &TransferFunction<T> {
    type Output = TransferFunction<T>;
    fn mul(self, rhs: &TransferFunction<T>) -> TransferFunction<T> {
        TransferFunction {
            numerator: &self.numerator * &rhs.numerator,
            denominator: &self.denominator * &rhs.denominator,
        }
        .reduced()
    }
}

impl<T: Float + AddAssign> Mul for TransferFunction<T> {
    type Output = TransferFunction<T>;
    fn mul(self, rhs: TransferFunction<T>) -> TransferFunction<T> {
        &self * &rhs
    }
}

impl<T: Float + AddAssign> Add for &TransferFunction<T> {
    type Output = TransferFunction<T>;
    fn add(self, rhs: &TransferFunction<T>) -> TransferFunction<T> {
        // n1/d1 + n2/d2 = (n1*d2 + n2*d1) / (d1*d2); `reduced()` then cancels any factor
        // shared by d1 and d2 (and any other common numerator/denominator roots), which is
        // equivalent to reducing to a common denominator first.
        let n1d2 = &self.numerator * &rhs.denominator;
        let n2d1 = &self.denominator * &rhs.numerator;
        let numerator = &n1d2 + &n2d1;
        let denominator = &self.denominator * &rhs.denominator;
        TransferFunction { numerator, denominator }.reduced()
    }
}

impl<T: Float + AddAssign> Add for TransferFunction<T> {
    type Output = TransferFunction<T>;
    fn add(self, rhs: TransferFunction<T>) -> TransferFunction<T> {
        &self + &rhs
    }
}
