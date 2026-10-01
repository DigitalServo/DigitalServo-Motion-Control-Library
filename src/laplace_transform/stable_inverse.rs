//! Stable inverse of a continuous-time system by the bilateral (two-sided) Laplace transform.
//!
//! A system `G(s)` with zeros in the right half-plane (nonminimum phase) has an inverse
//! `G^-1(s)` with unstable poles. Choosing the region of convergence that contains the
//! imaginary axis, the inverse Laplace transform of each partial fraction becomes
//!
//! ```text
//! r / (s - p)^k  <->   r t^(k-1) / (k-1)! e^(p t) 1(t)     (Re p < 0, causal)
//!                 <->  -r t^(k-1) / (k-1)! e^(p t) 1(-t)    (Re p > 0, anti-causal)
//! ```
//!
//! so the impulse response `h(t)` is bounded (it decays for both `t -> ∞` and `t -> -∞`)
//! at the cost of being non-causal. The polynomial part `Q(s)` gives impulses `δ^(n)(t)`.
//! No ROC contains the imaginary axis when `G^-1(s)` has a pole on it (`G(s)` has a zero on it).

use super::inverse_laplace::{eval_terms, format_time_terms};
use crate::{Continuous, PartialFraction, PoleTerm, Polynomial, TransferFunction};
use num_traits::Float;
use std::ops::AddAssign;
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum StableInverseError {
    #[error("The system is identically zero, so it has no inverse")]
    ZeroSystem,

    #[error(
        "Pole on the imaginary axis at {re} + j{im} (a zero of the original system, for `stable_inverse`): \
         no region of convergence contains the imaginary axis"
    )]
    PoleOnImaginaryAxis { re: f64, im: f64 },

    #[error("Pole at {re} + j{im} lies on the line Re s = {sigma} that must be inside the region of convergence")]
    PoleOnAbscissa { re: f64, im: f64, sigma: f64 },

    #[error(
        "No region of convergence: a causal pole (Re = {causal_re}) lies to the right of \
         an anti-causal pole (Re = {anticausal_re})"
    )]
    NoRegionOfConvergence { causal_re: f64, anticausal_re: f64 },

    #[error(
        "Reference trajectory is not smooth enough: state x{state} of the reference contains impulses \
         (the trajectory needs more continuous derivatives for this plant's relative degree)"
    )]
    NotSmoothEnough { state: usize },
}

/// `h(t)` split by the sign of `t`. `causal + anticausal` is the partial-fraction expansion of
/// the inverted system, as-is (residues are not negated here).
#[derive(Clone, Debug)]
pub struct StableInverse<T> {
    /// Polynomial part (impulses) and stable poles (`Re p < 0`): `h(t)` for `t >= 0`.
    pub causal: PartialFraction<T>,
    /// Unstable poles (`Re p > 0`); `direct` is empty. `h(t)` for `t < 0` is minus their
    /// causal inverse transforms.
    pub anticausal: PartialFraction<T>,
}

impl<T: Float + AddAssign> TransferFunction<T, Continuous> {
    /// Stable (non-causal) inverse of `G(s)`: the bilateral inverse Laplace transform of
    /// `1 / G(s)` whose region of convergence contains the imaginary axis.
    pub fn stable_inverse(&self) -> Result<StableInverse<T>, StableInverseError> {
        if self.numerator.iter().all(|c| c.is_zero()) {
            return Err(StableInverseError::ZeroSystem);
        }
        let inverse = TransferFunction::<T, Continuous>::from_polynomials(
            self.denominator.clone(),
            self.numerator.clone(),
        );
        StableInverse::bilateral(inverse.partial_fraction())
    }
}

impl<T: Float> StableInverse<T> {
    /// Bilateral inverse Laplace transform of `X(s)` (given as its partial-fraction expansion)
    /// with the region of convergence containing the imaginary axis.
    pub fn bilateral(pf: PartialFraction<T>) -> Result<Self, StableInverseError> {
        if let Some(term) = pf.terms.iter().find(|term| term.pole.re.is_zero()) {
            return Err(StableInverseError::PoleOnImaginaryAxis {
                re: to_f64(term.pole.re),
                im: to_f64(term.pole.im),
            });
        }
        Self::bilateral_with_abscissa(pf, T::zero())
    }

    /// Bilateral inverse Laplace transform of `X(s)` with the region of convergence containing
    /// the line `Re s = sigma`: poles with `Re p < sigma` are causal, `Re p > sigma` anti-causal.
    pub fn bilateral_with_abscissa(pf: PartialFraction<T>, sigma: T) -> Result<Self, StableInverseError> {
        if let Some(term) = pf.terms.iter().find(|term| term.pole.re == sigma) {
            return Err(StableInverseError::PoleOnAbscissa {
                re: to_f64(term.pole.re),
                im: to_f64(term.pole.im),
                sigma: to_f64(sigma),
            });
        }
        let (stable, unstable): (Vec<PoleTerm<T>>, Vec<PoleTerm<T>>) =
            pf.terms.into_iter().partition(|term| term.pole.re < sigma);
        Ok(Self {
            causal: PartialFraction { direct: pf.direct, terms: stable },
            anticausal: PartialFraction { direct: Polynomial(vec![]), terms: unstable },
        })
    }

    /// `h(t)` for any `t` (two-sided). Impulse terms `δ^(n)(t)` from `causal.direct` are
    /// not included (they vanish for `t != 0`).
    pub fn impulse_response(&self, t: T) -> T {
        if t < T::zero() {
            -eval_terms(&self.anticausal.terms, t)
        } else {
            eval_terms(&self.causal.terms, t)
        }
    }

    /// `h(t)` as a closure, e.g. `let h = inv.impulse_function(); h(-0.5)`. Same as
    /// `impulse_response`; the closure owns a copy, so it may outlive `self`.
    pub fn impulse_function(&self) -> impl Fn(T) -> T + use<T>
    where
        T: 'static,
    {
        let inv = self.clone();
        move |t| inv.impulse_response(t)
    }

    /// Displayable `h(t)`, e.g. `println!("{:.3}", inv.time_domain())`.
    pub fn time_domain(&self) -> StableInverseTimeDomain<'_, T> {
        StableInverseTimeDomain(self)
    }
}

fn to_f64<T: Float>(x: T) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}

/// `Display` wrapper for the two-sided time-domain expression of a `StableInverse`.
pub struct StableInverseTimeDomain<'a, T>(&'a StableInverse<T>);

impl<T: Float + std::fmt::Display> std::fmt::Display for StableInverseTimeDomain<'_, T> {
    /// e.g.
    /// ```text
    /// h(t) = δ(t) + 0.5 * exp(-2t)    (t >= 0)
    /// h(t) = -3 * exp(1t)             (t < 0)
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prec = f.precision();
        let negated = PartialFraction {
            direct: Polynomial(vec![]),
            terms: self
                .0
                .anticausal
                .terms
                .iter()
                .map(|term| PoleTerm { pole: term.pole, residues: term.residues.iter().map(|&r| -r).collect() })
                .collect(),
        };
        let causal = format_time_terms(&self.0.causal, prec);
        let anticausal = format_time_terms(&negated, prec);
        let width = causal.chars().count().max(anticausal.chars().count());
        writeln!(f, "h(t) = {:<width$}    (t >= 0)", causal)?;
        write!(f, "h(t) = {:<width$}    (t < 0)", anticausal)
    }
}
