//! Time-domain expression of a partial-fraction expansion as a real quasi-polynomial.

use crate::system::{fmt_num, push_term};
use crate::{PartialFraction, PoleTerm};
use num_traits::Float;
use serde::Serialize;

/// `e^(σt) [ P(t) cos(ωt) + Q(t) sin(ωt) ]` with real polynomials `P`, `Q` in `t`.
///
/// One mode covers a constant (`σ = ω = 0`), an exponential (`ω = 0`), powers of `t`
/// (`deg P > 0`), and (damped) oscillations (`ω > 0`), as well as their products.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TimeMode<T> {
    /// Exponential rate `σ` (real part of the pole).
    pub sigma: T,
    /// Angular frequency `ω >= 0` (imaginary part of the pole). `sin` is empty when `ω = 0`.
    pub omega: T,
    /// `cos[k]` is the coefficient of `t^k cos(ωt)` (ascending order).
    pub cos: Vec<T>,
    /// `sin[k]` is the coefficient of `t^k sin(ωt)` (ascending order).
    pub sin: Vec<T>,
}

/// Time-domain counterpart of `PartialFraction` (right-sided signal, `t >= 0`):
///
/// ```text
/// x(t) = Σ_n impulses[n] δ^(n)(t) + Σ_mode e^(σt) [ P(t) cos(ωt) + Q(t) sin(ωt) ]
/// ```
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TimeExpression<T> {
    /// `impulses[n]` is the coefficient of `δ^(n)(t)` (ascending order).
    pub impulses: Vec<T>,
    /// Modes with distinct `(σ, ω)`, in the order of the poles in the expansion.
    pub modes: Vec<TimeMode<T>>,
}

impl<T: Float> TimeMode<T> {
    /// Highest power of `t`.
    pub fn degree(&self) -> usize {
        self.cos.len().max(self.sin.len()).saturating_sub(1)
    }

    /// `ω != 0`.
    pub fn is_oscillatory(&self) -> bool {
        !self.omega.is_zero()
    }

    /// The mode for any `t` (no step function applied).
    pub fn eval(&self, t: T) -> T {
        let horner = |c: &[T]| c.iter().rev().fold(T::zero(), |acc, &a| acc * t + a);
        let mut v = horner(&self.cos);
        if self.is_oscillatory() {
            let (s, c) = (self.omega * t).sin_cos();
            v = v * c + horner(&self.sin) * s;
        }
        if self.sigma.is_zero() {
            v
        } else {
            v * (self.sigma * t).exp()
        }
    }
}

impl<T: Float> TimeExpression<T> {
    /// The expression for any `t` (no step function applied); impulses are not included.
    pub fn eval(&self, t: T) -> T {
        self.modes.iter().fold(T::zero(), |acc, m| acc + m.eval(t))
    }
}

impl<T: Float> PartialFraction<T> {
    /// Time-domain expression `x(t)` (`t >= 0`) as real modes; see `TimeExpression`.
    ///
    /// Each pole contributes `Re(c_k e^(pt)) t^k` with `c_k = r_{p,k+1} / k!`, i.e. with `p = σ + jω`
    ///
    /// ```text
    /// cos[k] += Re c_k,   sin[k] -= Im c_k      (ω > 0)
    /// cos[k] += Re c_k,   sin[k] += Im c_k      (ω < 0, folded to |ω|)
    /// ```
    ///
    /// so a conjugate pair `(p, conj p)` gives `cos[k] = 2 Re c_k`, `sin[k] = -2 Im c_k`. Poles within a
    /// relative distance of 1e-4 (after folding `ω` to `|ω|`) share a mode with the averaged `(σ, ω)`.
    /// This needs no exact conjugate pairing: the sum is what `time_response` evaluates.
    pub fn time_expression(&self) -> TimeExpression<T> {
        let rel_tol = T::from(1e-4).unwrap();
        let mut modes: Vec<TimeMode<T>> = Vec::new();
        // number of poles merged into each mode, for averaging (σ, ω)
        let mut counts: Vec<usize> = Vec::new();
        for term in &self.terms {
            let (sigma, omega) = (term.pole.re, term.pole.im.abs());
            let (cos, sin) = real_coefficients(term);
            let found = modes.iter().position(|m| {
                let d = (m.sigma - sigma).hypot(m.omega - omega);
                d <= rel_tol * sigma.hypot(omega)
            });
            match found {
                Some(i) => {
                    let m = &mut modes[i];
                    let n = T::from(counts[i]).unwrap();
                    m.sigma = (m.sigma * n + sigma) / (n + T::one());
                    m.omega = (m.omega * n + omega) / (n + T::one());
                    add_into(&mut m.cos, &cos);
                    add_into(&mut m.sin, &sin);
                    counts[i] += 1;
                }
                None => {
                    modes.push(TimeMode { sigma, omega, cos, sin });
                    counts.push(1);
                }
            }
        }
        for m in &mut modes {
            if m.omega.is_zero() {
                m.sin.clear();
            }
            trim_trailing_zeros(&mut m.cos);
            trim_trailing_zeros(&mut m.sin);
        }
        modes.retain(|m| !(m.cos.is_empty() && m.sin.is_empty()));

        TimeExpression { impulses: self.direct.iter().rev().copied().collect(), modes }
    }
}

/// `(cos, sin)` coefficients of one pole term, with `ω` folded to `|ω|`.
fn real_coefficients<T: Float>(term: &PoleTerm<T>) -> (Vec<T>, Vec<T>) {
    let sign = if term.pole.im < T::zero() { T::one() } else { -T::one() };
    let mut factorial = T::one();
    term.residues
        .iter()
        .enumerate()
        .map(|(k, &r)| {
            if k > 0 {
                factorial = factorial * T::from(k).unwrap();
            }
            let c = r / factorial;
            (c.re, sign * c.im)
        })
        .unzip()
}

fn add_into<T: Float>(acc: &mut Vec<T>, x: &[T]) {
    if acc.len() < x.len() {
        acc.resize(x.len(), T::zero());
    }
    for (a, &b) in acc.iter_mut().zip(x) {
        *a = *a + b;
    }
}

fn trim_trailing_zeros<T: Float>(c: &mut Vec<T>) {
    while c.last().is_some_and(|x| x.is_zero()) {
        c.pop();
    }
}

impl<T: Float> std::ops::Neg for TimeExpression<T> {
    type Output = Self;

    fn neg(self) -> Self {
        let neg = |c: Vec<T>| c.into_iter().map(|x| -x).collect();
        TimeExpression {
            impulses: neg(self.impulses),
            modes: self
                .modes
                .into_iter()
                .map(|m| TimeMode { sigma: m.sigma, omega: m.omega, cos: neg(m.cos), sin: neg(m.sin) })
                .collect(),
        }
    }
}

/// How the oscillatory part `a cos(ωt) + b sin(ωt)` of a mode is written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrigForm {
    /// `a cos(ωt) + b sin(ωt)`
    #[default]
    CosSin,
    /// `A cos(ωt + φ)` with `A = √(a² + b²)`, `φ = atan2(-b, a)`
    Cos,
    /// `A sin(ωt + φ)` with `A = √(a² + b²)`, `φ = atan2(a, b)`
    Sin,
}

impl<T: Float> TimeMode<T> {
    /// Amplitude `A >= 0` and phase `φ ∈ (-π, π]` \[rad\] of the `t^k` coefficient written as
    /// `A cos(ωt + φ)` (`TrigForm::Cos`) or `A sin(ωt + φ)` (`TrigForm::Sin`).
    /// `None` for `TrigForm::CosSin`.
    pub fn amplitude_phase(&self, k: usize, form: TrigForm) -> Option<(T, T)> {
        let a = self.cos.get(k).copied().unwrap_or_else(T::zero);
        let b = self.sin.get(k).copied().unwrap_or_else(T::zero);
        let phase = match form {
            TrigForm::CosSin => return None,
            TrigForm::Cos => (-b).atan2(a),
            TrigForm::Sin => a.atan2(b),
        };
        Some((a.hypot(b), phase))
    }
}

impl<T: Float + std::fmt::Display> TimeExpression<T> {
    /// The expression as a string (`prec`: digits after the decimal point).
    pub(super) fn format(&self, form: TrigForm, prec: Option<usize>) -> String {
        let mut out = String::new();

        for (n, &c) in self.impulses.iter().enumerate().rev() {
            let body = match n {
                0 => "δ(t)".to_string(),
                n => format!("δ^({})(t)", n),
            };
            push_term(&mut out, c, &body, prec);
        }

        for m in &self.modes {
            let exp = if m.sigma.is_zero() { None } else { Some(format!("exp({}t)", fmt_num(m.sigma, prec))) };
            let w = fmt_num(m.omega, prec);
            let zero = T::zero();
            for k in 0..=m.degree() {
                let a = m.cos.get(k).copied().unwrap_or(zero);
                let b = m.sin.get(k).copied().unwrap_or(zero);
                let mut factors: Vec<String> = Vec::new();
                match k {
                    0 => {}
                    1 => factors.push("t".to_string()),
                    _ => factors.push(format!("t^{}", k)),
                }
                factors.extend(exp.clone());
                if !m.is_oscillatory() {
                    push_term(&mut out, a, &factors.join(" * "), prec);
                    continue;
                }
                if a.is_zero() && b.is_zero() {
                    continue;
                }
                match m.amplitude_phase(k, form) {
                    None => {
                        let mut trig = String::new();
                        push_term(&mut trig, a, &format!("cos({}t)", w), prec);
                        push_term(&mut trig, b, &format!("sin({}t)", w), prec);
                        factors.push(format!("({})", trig));
                        push_term(&mut out, T::one(), &factors.join(" * "), prec);
                    }
                    Some((amplitude, phase)) => {
                        let func = if form == TrigForm::Sin { "sin" } else { "cos" };
                        let mut arg = format!("{}t", w);
                        if !phase.is_zero() {
                            let sign = if phase.is_sign_negative() { '-' } else { '+' };
                            arg = format!("{} {} {}", arg, sign, fmt_num(phase.abs(), prec));
                        }
                        factors.push(format!("{}({})", func, arg));
                        push_term(&mut out, amplitude, &factors.join(" * "), prec);
                    }
                }
            }
        }
        if out.is_empty() {
            out = fmt_num(T::zero(), prec);
        }
        out
    }
}

/// `Display` wrapper `x(t) = ...` of a `TimeExpression`, with the oscillations written in `form`,
/// e.g. `println!("{:.3}", pf.time_domain().trig_form(TrigForm::Sin))` or
/// `TimeDomain::from(expr).trig_form(TrigForm::Cos)`.
pub struct TimeDomain<T> {
    /// The expression.
    pub expression: TimeExpression<T>,
    /// How oscillations are written.
    pub form: TrigForm,
}

impl<T> From<TimeExpression<T>> for TimeDomain<T> {
    /// `TrigForm::CosSin`.
    fn from(expression: TimeExpression<T>) -> Self {
        Self { expression, form: TrigForm::default() }
    }
}

impl<T> TimeDomain<T> {
    /// Write oscillations in `form`.
    pub fn trig_form(self, form: TrigForm) -> Self {
        Self { form, ..self }
    }
}

impl<T: Float + std::fmt::Display> std::fmt::Display for TimeDomain<T> {
    /// e.g. `x(t) = 2 * exp(-1t) + 3 * t * exp(-1t) + exp(-1t) * (2 * cos(2t) - 1 * sin(2t))`
    /// (`TrigForm::CosSin`), `x(t) = 2.236 * exp(-1t) * sin(2t + 2.034)` (`TrigForm::Sin`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "x(t) = {}", self.expression.format(self.form, f.precision()))
    }
}

impl<T: Float + std::fmt::Display> std::fmt::Display for TimeExpression<T> {
    /// e.g. `2 * exp(-1t) + 3 * t * exp(-1t) + exp(-1t) * (2 * cos(2t) - 1 * sin(2t))`
    /// (`TrigForm::CosSin`; see `TimeDomain` for the amplitude-phase forms).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.format(TrigForm::CosSin, f.precision()))
    }
}
