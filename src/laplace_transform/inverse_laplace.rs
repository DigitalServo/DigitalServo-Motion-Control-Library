//! Inverse Laplace transform of a partial-fraction expansion (right-sided / causal signal).

use super::TimeDomain;
use crate::{Continuous, PartialFraction, PoleTerm, TransferFunction};
use num_complex::Complex;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

impl<T: Float + AddAssign> TransferFunction<T, Continuous> {
    /// Inverse Laplace transform `x(t)` as a closure (see `PartialFraction::time_function`),
    /// e.g. `let x = tf!("1 / (s + 1)").inverse_laplace(); x(0.5)`.
    ///
    /// ```text
    /// X(s) = Q(s) + Σ_p Σ_{k=1..m_p} r_{p,k} / (s - p)^k
    /// x(t) = Σ_n q_n δ^(n)(t) + Σ_p Σ_{k=1..m_p} r_{p,k} t^(k-1) / (k-1)! e^(p t)      (t >= 0)
    /// ```
    pub fn inverse_laplace(&self) -> impl Fn(T) -> T + use<T>
    where
        T: 'static,
    {
        self.partial_fraction().time_function()
    }
}

impl<T: Float> PartialFraction<T> {
    /// `x(t)` for the right-sided signal: 0 for `t < 0`. Impulse terms `δ^(n)(t)` coming from
    /// the polynomial part `direct` are not included (they vanish for `t != 0`).
    pub fn time_response(&self, t: T) -> T {
        if t < T::zero() {
            return T::zero();
        }
        eval_terms(&self.terms, t)
    }

    /// `x(t)` as a closure, e.g. `let x = pf.time_function(); x(0.5)`. Same as `time_response`
    /// (impulse terms excluded); the closure owns the real modes of `time_expression`
    /// (conjugate pairs merged), so it may outlive `self`.
    pub fn time_function(&self) -> impl Fn(T) -> T + use<T>
    where
        T: 'static,
    {
        let expr = self.time_expression();
        move |t| if t < T::zero() { T::zero() } else { expr.eval(t) }
    }

    /// Displayable `x(t)`, e.g. `println!("{:.3}", pf.time_domain())`, or with the oscillations
    /// as amplitude and phase, `pf.time_domain().trig_form(TrigForm::Cos)`.
    pub fn time_domain(&self) -> TimeDomain<T> {
        TimeDomain::from(self.time_expression())
    }
}

/// `Σ_p Σ_k r_{p,k} t^(k-1) / (k-1)! e^(p t)` for any `t` (no step function applied).
pub(super) fn eval_terms<T: Float>(terms: &[PoleTerm<T>], t: T) -> T {
    let mut sum = Complex::zero();
    for term in terms {
        let exp_pt = (term.pole * t).exp();
        // t^k / k!
        let mut basis = T::one();
        for (k, &r) in term.residues.iter().enumerate() {
            if k > 0 {
                basis = basis * t / T::from(k).unwrap();
            }
            sum = sum + r * exp_pt * basis;
        }
    }
    sum.re
}
