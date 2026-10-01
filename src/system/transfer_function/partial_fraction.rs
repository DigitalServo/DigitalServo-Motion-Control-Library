//! Partial-fraction expansion of `X(s) = N(s) / D(s)` by the Heaviside expansion theorem,
//! and the inverse Laplace transform `x(t)` (right-sided / causal signal) built from it.
//!
//! ```text
//! X(s) = Q(s) + Σ_p Σ_{k=1..m_p} r_{p,k} / (s - p)^k
//! x(t) = Σ_n q_n δ^(n)(t) + Σ_p Σ_{k=1..m_p} r_{p,k} t^(k-1) / (k-1)! e^(p t)      (t >= 0)
//! ```
//! For a pole `p` of multiplicity `m`, `r_{p,k} = G^(m-k)(p) / (m-k)!` with `G(s) = (s - p)^m X(s)`.
//! The derivatives are obtained exactly as Taylor coefficients of `G` around `p`
//! (synthetic division + power-series division), not by numerical differentiation.

use super::TransferFunction;
use crate::{dka_method, vieta_formula, Continuous, Polynomial};
use num_complex::Complex;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// Terms of one (possibly repeated) pole.
#[derive(Clone, Debug)]
pub struct PoleTerm<T> {
    pub pole: Complex<T>,
    /// `residues[k]` is the coefficient of `1 / (s - pole)^(k + 1)`; its length is the multiplicity.
    pub residues: Vec<Complex<T>>,
}

impl<T> PoleTerm<T> {
    pub fn multiplicity(&self) -> usize {
        self.residues.len()
    }
}

/// `X(s) = direct(s) + Σ terms`.
#[derive(Clone, Debug)]
pub struct PartialFraction<T> {
    /// Polynomial part `Q(s)` (descending order). Empty when `X(s)` is strictly proper.
    pub direct: Polynomial<T>,
    pub terms: Vec<PoleTerm<T>>,
}

impl<T: Float + AddAssign> TransferFunction<T, Continuous> {
    /// Partial-fraction expansion by the Heaviside expansion theorem (repeated poles supported).
    /// Poles closer than a relative distance of 1e-4 are treated as one repeated pole.
    pub fn partial_fraction(&self) -> PartialFraction<T> {
        self.partial_fraction_with_tolerance(T::from(1e-4).unwrap())
    }

    /// Inverse Laplace transform `x(t)` as a closure (see `PartialFraction::time_function`),
    /// e.g. `let x = tf!("1 / (s + 1)").inverse_laplace(); x(0.5)`.
    pub fn inverse_laplace(&self) -> impl Fn(T) -> T + use<T>
    where
        T: 'static,
    {
        self.partial_fraction().time_function()
    }

    /// Same as `partial_fraction`, but with an explicit relative tolerance for grouping
    /// numerically found roots into repeated poles.
    pub fn partial_fraction_with_tolerance(&self, rel_tol: T) -> PartialFraction<T> {
        let numer = trim_leading_zeros(&self.numerator);
        let denom = trim_leading_zeros(&self.denominator);
        assert!(!denom.is_empty(), "partial_fraction: denominator is zero");

        let (direct, remainder) = polynomial_division(&numer, &denom);
        let gain = Complex::from(denom[0]);
        let remainder: Vec<Complex<T>> = remainder.iter().map(|&c| Complex::from(c)).collect();

        let denom_complex = Polynomial(denom.iter().map(|&c| Complex::from(c)).collect());
        let roots = dka_method(&denom_complex).unwrap_or_default();
        let poles = group_roots(&denom_complex, &roots, rel_tol);

        let terms = poles
            .iter()
            .enumerate()
            .map(|(i, &(p, m))| {
                // B(s) = D(s) / (s - p)^m, so that G(s) = (s - p)^m X(s) = R(s) / B(s).
                let others: Vec<Complex<T>> = poles
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .flat_map(|(_, &(q, mq))| std::iter::repeat_n(q, mq))
                    .collect();
                let b: Vec<Complex<T>> = vieta_formula(&others).0.iter().map(|&c| c * gain).collect();
                PoleTerm { pole: p, residues: principal_part(&remainder, &b, p, m) }
            })
            .collect();

        PartialFraction { direct, terms }
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

fn trim_leading_zeros<T: Float>(p: &Polynomial<T>) -> Vec<T> {
    p.iter().copied().skip_while(|c| c.is_zero()).collect()
}

/// `numer = quotient * denom + remainder` for descending-order coefficients
/// (`denom` must have a nonzero leading coefficient). `remainder` has `denom.len() - 1` entries.
fn polynomial_division<T: Float>(numer: &[T], denom: &[T]) -> (Polynomial<T>, Vec<T>) {
    let rem_len = denom.len() - 1;
    if numer.len() < denom.len() {
        let mut remainder = vec![T::zero(); rem_len - numer.len()];
        remainder.extend_from_slice(numer);
        return (Polynomial(vec![]), remainder);
    }
    let mut work = numer.to_vec();
    let quotient_len = numer.len() - rem_len;
    let mut quotient = vec![T::zero(); quotient_len];
    for i in 0..quotient_len {
        let c = work[i] / denom[0];
        quotient[i] = c;
        for (j, &d) in denom.iter().enumerate() {
            work[i + j] = work[i + j] - c * d;
        }
    }
    (Polynomial(quotient), work.split_off(quotient_len))
}

/// Group numerically found roots into `(pole, multiplicity)`. A repeated root of multiplicity m
/// is found as a small cluster (spread ~ eps^(1/m)), so roots within `rel_tol` are merged.
/// The cluster mean is then refined by Newton's method on `D^(m-1)(s)`, for which the pole is
/// a simple root. Poles nearly on the real / imaginary axis are snapped onto it.
pub(super) fn group_roots<T: Float>(denom: &Polynomial<Complex<T>>, roots: &[Complex<T>], rel_tol: T) -> Vec<(Complex<T>, usize)> {
    // (sum of members, count)
    let mut groups: Vec<(Complex<T>, usize)> = Vec::new();
    for &r in roots {
        let found = groups.iter_mut().find(|(sum, n)| {
            let center = *sum / T::from(*n).unwrap();
            (center - r).norm() <= rel_tol * center.norm().max(T::one())
        });
        match found {
            Some((sum, n)) => {
                *sum = *sum + r;
                *n += 1;
            }
            None => groups.push((r, 1)),
        }
    }
    groups
        .into_iter()
        .map(|(sum, m)| {
            let mut p = sum / T::from(m).unwrap();
            if m > 1 {
                p = refine_repeated_root(denom, p, m);
            }
            let scale = rel_tol * p.norm().max(T::one());
            if p.im.abs() <= scale {
                p.im = T::zero();
            }
            if p.re.abs() <= scale {
                p.re = T::zero();
            }
            (p, m)
        })
        .collect()
}

/// Newton's method on `D^(m-1)(s)`, whose root at a pole of multiplicity m is simple.
/// With Taylor coefficients `c_j = D^(j)(p) / j!`, the step is `c_{m-1} / (m c_m)`.
fn refine_repeated_root<T: Float>(denom: &Polynomial<Complex<T>>, mut p: Complex<T>, m: usize) -> Complex<T> {
    let eps = T::epsilon();
    for _ in 0..20 {
        let c = taylor_coefficients(&denom.0, p, m + 1);
        if c[m].is_zero() {
            break;
        }
        let step = c[m - 1] / (c[m] * T::from(m).unwrap());
        p = p - step;
        if step.norm() <= eps * p.norm().max(T::one()) {
            break;
        }
    }
    p
}

/// Residues of `R(s) / ((s - p)^m B(s))` at `p` (`B(p) != 0`, descending `numer` / `rest`):
/// element k is the coefficient of `1 / (s - p)^(k + 1)`. With `G = R / B`, these are
/// `G^(m-1-k)(p) / (m-1-k)!`, obtained by power-series division of Taylor coefficients.
pub(super) fn principal_part<T: Float>(
    numer: &[Complex<T>],
    rest: &[Complex<T>],
    p: Complex<T>,
    m: usize,
) -> Vec<Complex<T>> {
    let r_taylor = taylor_coefficients(numer, p, m);
    let b_taylor = taylor_coefficients(rest, p, m);

    // Power-series division: g_j = G^(j)(p) / j!
    let mut g = vec![Complex::zero(); m];
    for j in 0..m {
        let mut acc = r_taylor[j];
        for l in 1..=j {
            acc = acc - b_taylor[l] * g[j - l];
        }
        g[j] = acc / b_taylor[0];
    }
    // g_j is the coefficient of 1 / (s - p)^(m - j)
    g.reverse();
    g
}

/// First `count` ascending Taylor coefficients of `P(x0 + h)` (descending `coeffs`),
/// i.e. `P^(j)(x0) / j!` for j = 0..count, by repeated synthetic division by `(s - x0)`.
pub(super) fn taylor_coefficients<T: Float>(coeffs: &[Complex<T>], x0: Complex<T>, count: usize) -> Vec<Complex<T>> {
    let mut work = coeffs.to_vec();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut quotient = Vec::with_capacity(work.len());
        let mut acc = Complex::zero();
        for &a in &work {
            acc = acc * x0 + a;
            quotient.push(acc);
        }
        out.push(quotient.pop().unwrap_or_else(Complex::zero));
        work = quotient;
    }
    out
}

fn fmt_num<T: Float + std::fmt::Display>(c: T, precision: Option<usize>) -> String {
    match precision {
        Some(p) => format!("{:.*}", p, c),
        None => c.to_string(),
    }
}

fn fmt_complex<T: Float + std::fmt::Display>(c: Complex<T>, precision: Option<usize>) -> String {
    if c.im.is_zero() {
        return fmt_num(c.re, precision);
    }
    let sign = if c.im.is_sign_negative() { '-' } else { '+' };
    format!("({} {} j{})", fmt_num(c.re, precision), sign, fmt_num(c.im.abs(), precision))
}

/// Append `coef * body` to a sum, folding the sign of `coef` into the separator.
fn push_term<T: Float + std::fmt::Display>(out: &mut String, coef: T, body: &str, precision: Option<usize>) {
    push_term_with(out, coef, " * ", body, precision);
}

/// Append `coef{op}body` (`op` = `" * "` or `" / "`) to a sum, folding the sign of `coef` into the separator.
/// A unit coefficient is omitted for `" * "`.
fn push_term_with<T: Float + std::fmt::Display>(
    out: &mut String,
    coef: T,
    op: &str,
    body: &str,
    precision: Option<usize>,
) {
    if coef.is_zero() {
        return;
    }
    let negative = coef.is_sign_negative();
    match (out.is_empty(), negative) {
        (true, true) => out.push('-'),
        (true, false) => {}
        (false, true) => out.push_str(" - "),
        (false, false) => out.push_str(" + "),
    }
    let abs = coef.abs();
    if body.is_empty() {
        out.push_str(&fmt_num(abs, precision));
    } else if abs == T::one() && op == " * " {
        out.push_str(body);
    } else {
        out.push_str(&format!("{}{}{}", fmt_num(abs, precision), op, body));
    }
}

impl<T: Float + std::fmt::Display> std::fmt::Display for PartialFraction<T> {
    /// e.g. `s + 1 + 2 / (s + 1) + 3 / (s + 1)^2` (complex poles are printed as `(a + jb)`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prec = f.precision();
        let mut out = String::new();
        let degree = self.direct.len().saturating_sub(1);
        for (i, &c) in self.direct.iter().enumerate() {
            let body = match degree - i {
                0 => String::new(),
                1 => "s".to_string(),
                n => format!("s^{}", n),
            };
            push_term(&mut out, c, &body, prec);
        }
        for term in &self.terms {
            let p = term.pole;
            let factor = if p.is_zero() {
                "s".to_string()
            } else if p.im.is_zero() {
                let sign = if p.re.is_sign_negative() { '+' } else { '-' };
                format!("(s {} {})", sign, fmt_num(p.re.abs(), prec))
            } else {
                format!("(s - {})", fmt_complex(p, prec))
            };
            for (k, &r) in term.residues.iter().enumerate() {
                let power = if k == 0 { factor.clone() } else { format!("{}^{}", factor, k + 1) };
                if r.im.is_zero() {
                    push_term_with(&mut out, r.re, " / ", &power, prec);
                } else if !r.is_zero() {
                    if !out.is_empty() {
                        out.push_str(" + ");
                    }
                    out.push_str(&format!("{} / {}", fmt_complex(r, prec), power));
                }
            }
        }
        if out.is_empty() {
            out = fmt_num(T::zero(), prec);
        }
        write!(f, "{}", out)
    }
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
