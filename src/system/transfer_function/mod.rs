use crate::{dka_method, vieta_formula, Continuous, Discrete, Polynomial};
use num_complex::Complex;
use num_traits::Float;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul};

mod display;
mod parser;
pub use parser::TransferFunctionParseError;

/// Descending-order numerator / denominator polynomials. `D` tells whether they are in `s`
/// (`Continuous`, the default) or in `z` (`Discrete`), so the two cannot be mixed up.
#[derive(Clone, Debug)]
pub struct TransferFunction<T, D = Continuous> {
    pub numerator: Polynomial<T>,
    pub denominator: Polynomial<T>,
    _domain: PhantomData<D>,
}

#[derive(Clone, Debug)]
pub struct PzMap<T, D = Continuous> {
    pub poles: Vec<Complex<T>>,
    pub zeros: Vec<Complex<T>>,
    _domain: PhantomData<D>,
}

impl<T, D> TransferFunction<T, D> {
    /// Build from polynomials as-is (no pole-zero cancellation, unlike `continuous` / `discrete`).
    pub fn from_polynomials(numerator: Polynomial<T>, denominator: Polynomial<T>) -> Self {
        Self { numerator, denominator, _domain: PhantomData }
    }
}


impl<T: Float + AddAssign, D> TransferFunction<T, D> {
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
        Self::from_polynomials(
            reconstruct(numer_gain, &numer_roots),
            reconstruct(denom_gain, &denom_roots),
        )
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

impl<T: Float + AddAssign> TransferFunction<T, Continuous> {
    /// Continuous-time transfer function from descending-order coefficients in `s`.
    /// Common poles/zeros are cancelled (see `reduced`).
    pub fn continuous(numerator: &[T], denominator: &[T]) -> Self {
        Self::from_polynomials(Polynomial(numerator.to_vec()), Polynomial(denominator.to_vec())).reduced()
    }
}

impl<T: Float + AddAssign> TransferFunction<T, Discrete> {
    /// Discrete-time transfer function from descending-order coefficients in `z`.
    /// Common poles/zeros are cancelled (see `reduced`).
    pub fn discrete(numerator: &[T], denominator: &[T]) -> Self {
        Self::from_polynomials(Polynomial(numerator.to_vec()), Polynomial(denominator.to_vec())).reduced()
    }
}

impl<T: Float + AddAssign, D> TransferFunction<T, D> {
    pub fn pz_map(&self) -> PzMap<T, D> {
        let tf = self.reduced();

        let (denom, numer) = {
            let denom_complex = tf.denominator
                .iter()
                .map(|&x| Complex::from(x))
                .collect::<Vec<Complex<T>>>();

            let numer_complex = tf.numerator
                .iter()
                .map(|&x| Complex::from(x))
                .collect::<Vec<Complex<T>>>();

            (Polynomial(denom_complex), Polynomial(numer_complex))
        };

        PzMap {
            poles: dka_method(&denom).unwrap_or(vec![]),
            zeros: dka_method(&numer).unwrap_or(vec![]),
            _domain: PhantomData,
        }
    }
}
