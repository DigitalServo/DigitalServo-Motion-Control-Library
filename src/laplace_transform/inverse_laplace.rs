//! Inverse Laplace transform of a partial-fraction expansion (right-sided / causal signal).

use crate::system::{fmt_num, push_term};
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
    /// (impulse terms excluded); the closure owns a copy of the expansion, so it may outlive `self`.
    pub fn time_function(&self) -> impl Fn(T) -> T + use<T>
    where
        T: 'static,
    {
        let pf = self.clone();
        move |t| pf.time_response(t)
    }

    /// Displayable `x(t)`, e.g. `println!("{:.3}", pf.time_domain())`.
    pub fn time_domain(&self) -> TimeDomain<'_, T> {
        TimeDomain(self)
    }
}

/// `Display` wrapper for the time-domain expression of a `PartialFraction`.
pub struct TimeDomain<'a, T>(&'a PartialFraction<T>);

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

impl<T: Float + std::fmt::Display> std::fmt::Display for TimeDomain<'_, T> {
    /// e.g. `x(t) = 2 * exp(-1t) + 3 * t * exp(-1t) + exp(-1t) * (2 * cos(2t) - 1 * sin(2t))`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "x(t) = {}", format_time_terms(self.0, f.precision()))
    }
}

/// Time-domain expression of `pf` (impulses `δ^(n)(t)` for `direct`, exponentials for `terms`).
/// A conjugate pole pair is merged into `exp(σt) * (A cos(ωt) + B sin(ωt))`
/// (the pole with negative imaginary part is skipped, since X(s) has real coefficients).
pub(super) fn format_time_terms<T: Float + std::fmt::Display>(pf: &PartialFraction<T>, prec: Option<usize>) -> String {
    let mut out = String::new();

    // Polynomial part q_n s^n  <->  q_n δ^(n)(t)
    let degree = pf.direct.len().saturating_sub(1);
    for (i, &c) in pf.direct.iter().enumerate() {
        let body = match degree - i {
            0 => "δ(t)".to_string(),
            n => format!("δ^({})(t)", n),
        };
        push_term(&mut out, c, &body, prec);
    }

    for term in &pf.terms {
        let p = term.pole;
        if p.im < T::zero() {
            continue;
        }
        let exp = if p.re.is_zero() { None } else { Some(format!("exp({}t)", fmt_num(p.re, prec))) };
        let mut factorial = T::one();
        for (k, &r) in term.residues.iter().enumerate() {
            if k > 0 {
                factorial = factorial * T::from(k).unwrap();
            }
            let r = r / factorial;
            let mut factors: Vec<String> = Vec::new();
            match k {
                0 => {}
                1 => factors.push("t".to_string()),
                _ => factors.push(format!("t^{}", k)),
            }
            factors.extend(exp.clone());

            if p.im.is_zero() {
                push_term(&mut out, r.re, &factors.join(" * "), prec);
            } else {
                // r e^{pt} + conj(r) e^{conj(p)t} = e^{σt} (2Re(r) cos(ωt) - 2Im(r) sin(ωt))
                let two = T::from(2.0).unwrap();
                let (a, b) = (two * r.re, -two * r.im);
                if a.is_zero() && b.is_zero() {
                    continue;
                }
                let w = fmt_num(p.im, prec);
                let mut trig = String::new();
                push_term(&mut trig, a, &format!("cos({}t)", w), prec);
                push_term(&mut trig, b, &format!("sin({}t)", w), prec);
                factors.push(format!("({})", trig));
                push_term(&mut out, T::one(), &factors.join(" * "), prec);
            }
        }
    }
    if out.is_empty() {
        out = fmt_num(T::zero(), prec);
    }
    out
}
