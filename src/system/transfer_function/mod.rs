use crate::{dka_method, Continuous, Discrete, Polynomial};
use num_complex::Complex;
use num_traits::{Float, Zero};
use serde::Serialize;
use std::marker::PhantomData;
use std::ops::AddAssign;

mod display;
mod ops;
mod parser;
mod partial_fraction;
mod with_delay;
pub use partial_fraction::{PartialFraction, PoleTerm};
pub use with_delay::{SimulationError, TransferFunctionWithDelay};
pub(crate) use partial_fraction::{fmt_num, principal_part, push_term};

pub use parser::TransferFunctionParseError;
#[doc(hidden)]
pub use parser::{__detect_domain, __DomainTag, __SelectDomain};

/// Descending-order numerator / denominator polynomials. `D` tells whether they are in `s`
/// (`Continuous`, the default) or in `z` (`Discrete`), so the two cannot be mixed up.
#[derive(Clone, Debug, Serialize)]
pub struct TransferFunction<T, D = Continuous> {
    /// Numerator coefficients, descending order.
    pub numerator: Polynomial<T>,
    /// Denominator coefficients, descending order.
    pub denominator: Polynomial<T>,
    #[serde(skip)]
    _domain: PhantomData<D>,
}

/// Poles and zeros of a `TransferFunction` (see `TransferFunction::pz_map`).
#[derive(Clone, Debug, Serialize)]
pub struct PzMap<T, D = Continuous> {
    /// Roots of the denominator.
    pub poles: Vec<Complex<T>>,
    /// Roots of the numerator.
    pub zeros: Vec<Complex<T>>,
    #[serde(skip)]
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
    /// Common factors are removed by dividing the original coefficients by `(s - c)`
    /// (synthetic division), rather than rebuilding the polynomials from numerically found roots,
    /// so the remaining factors keep their coefficients (repeated roots included).
    ///
    /// Roots at the origin are known exactly from the trailing zero coefficients, so the common
    /// ones are cancelled first by dropping those zeros. Only the remaining roots go through the
    /// numerical root finder, where a root of high multiplicity would spread too widely to match.
    pub fn reduced_with_tolerance(&self, rel_tol: T) -> Self {
        let mut numer = trim_leading_zeros(&self.numerator);
        let mut denom = trim_leading_zeros(&self.denominator);
        let origin = trailing_zeros(&numer).min(trailing_zeros(&denom));
        let (numer_len, denom_len) = (numer.len(), denom.len());
        numer.truncate(numer_len - origin);
        denom.truncate(denom_len - origin);

        let numer_roots = grouped_roots(&numer);
        let denom_roots = grouped_roots(&denom);

        // (common root, how many times it cancels)
        let mut common: Vec<(Complex<T>, usize)> = Vec::new();
        for &(z, mz) in &numer_roots {
            let threshold = rel_tol * z.norm().max(T::one());
            if let Some(&(p, mp)) = denom_roots.iter().find(|&&(p, _)| (p - z).norm() <= threshold) {
                common.push(((z + p) / T::from(2.0).unwrap(), mz.min(mp)));
            }
        }
        if common.is_empty() {
            return Self::from_polynomials(numer, denom);
        }

        Self::from_polynomials(deflate(&numer, &common), deflate(&denom, &common))
    }
}

/// Drop leading zero coefficients; the zero polynomial becomes `[0]`.
fn trim_leading_zeros<T: Float>(p: &Polynomial<T>) -> Polynomial<T> {
    let coeffs: Vec<T> = p.iter().copied().skip_while(|c| c.is_zero()).collect();
    if coeffs.is_empty() { Polynomial(vec![T::zero()]) } else { Polynomial(coeffs) }
}

/// Multiplicity of the root at the origin: the number of trailing zero coefficients
/// (`0` for the zero polynomial, which has no well-defined roots).
fn trailing_zeros<T: Float>(p: &Polynomial<T>) -> usize {
    if p.iter().all(|c| c.is_zero()) {
        return 0;
    }
    p.iter().rev().take_while(|c| c.is_zero()).count()
}

/// Roots of a descending-order real polynomial as `(root, multiplicity)`. Clusters of a repeated
/// root (spread ~eps^(1/m)) are merged and refined (see `partial_fraction::group_roots`).
fn grouped_roots<T: Float>(p: &Polynomial<T>) -> Vec<(Complex<T>, usize)> {
    let tol = T::from(1e-4).unwrap();
    roots_with_multiplicity(p, tol, tol)
}

/// Roots of a descending-order real polynomial as `(root, multiplicity)`: roots within `cluster_tol`
/// are merged into a refined repeated root, and real / imaginary parts within `snap_tol` are set
/// to zero (both relative to `max(|root|, 1)`).
pub(crate) fn roots_with_multiplicity<T: Float>(p: &Polynomial<T>, cluster_tol: T, snap_tol: T) -> Vec<(Complex<T>, usize)> {
    let complex_poly = Polynomial(p.iter().map(|&c| Complex::from(c)).collect());
    let roots = dka_method(&complex_poly).unwrap_or_default();
    partial_fraction::group_roots(&complex_poly, &roots, cluster_tol, snap_tol)
}

/// Divide `p` by `Π (s - c)^k` with synthetic division, discarding the (round-off) remainders.
/// Common roots of a real polynomial come in conjugate pairs, so the result is real up to
/// round-off and its real part is returned.
fn deflate<T: Float>(p: &Polynomial<T>, factors: &[(Complex<T>, usize)]) -> Polynomial<T> {
    let mut work: Vec<Complex<T>> = p.iter().map(|&c| Complex::from(c)).collect();
    for &(c, k) in factors {
        for _ in 0..k {
            if work.len() <= 1 {
                break;
            }
            let mut acc = Complex::zero();
            let mut quotient = Vec::with_capacity(work.len());
            for &a in &work {
                acc = acc * c + a;
                quotient.push(acc);
            }
            quotient.pop();
            work = quotient;
        }
    }
    Polynomial(work.iter().map(|c| c.re).collect())
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
    /// Poles and zeros, after pole-zero cancellation (`reduced`).
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
