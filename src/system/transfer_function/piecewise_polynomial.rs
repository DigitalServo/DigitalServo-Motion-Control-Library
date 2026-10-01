//! Piecewise-polynomial signals, e.g. smooth rest-to-rest trajectories.
//!
//! `y(t) = 0` before `start`, then each piece `p_q(τ)` (τ = time since the piece started) for its
//! duration, then `tail(τ)` forever (τ = time since the last piece ended).
//!
//! The exact Laplace transform is a sum of delayed rationals (`laplace`): at each breakpoint the
//! jump `Δ_q(τ) = p_q(τ) - p_{q-1}(τ + T_{q-1}) = Σ c_m τ^m` starts, with `L[Δ_q] = Σ c_m m! / s^(m+1)`.
//! Evaluating `y` as that sum would subtract large, growing polynomials (∝ (t/T)^deg) long after
//! the move, so values and derivatives are always evaluated from the local piece instead.

use super::{LaplaceSignal, TransferFunction};
use crate::{Continuous, Polynomial};
use num_traits::Float;

/// Jump coefficients smaller than this (relative to the natural size of the adjacent pieces) are
/// round-off from the Taylor shift and are set to zero, so that continuity is detected exactly.
const JUMP_RELATIVE_TOLERANCE: f64 = 1e-9;

#[derive(Clone, Debug)]
pub struct PiecewisePolynomial<T> {
    /// Time at which the first piece starts (`y(t) = 0` before).
    pub start: T,
    /// `(duration, p)`, with `p` a descending polynomial in the time since the piece started.
    pub pieces: Vec<(T, Polynomial<T>)>,
    /// Descending polynomial in the time since the last piece ended, used from then on
    /// (e.g. `[distance]` to stay at rest).
    pub tail: Polynomial<T>,
}

/// One breakpoint of the jump decomposition.
#[derive(Clone, Debug)]
pub(super) struct Jump<T> {
    pub time: T,
    /// Ascending coefficients of `Δ(τ)` (trailing zeros removed, lowest ones may be exactly zero).
    pub coefficients: Vec<T>,
}

impl<T: Float> Jump<T> {
    /// Index of the lowest nonzero coefficient: `y^(m)` jumps here, lower derivatives are continuous.
    pub fn order(&self) -> usize {
        self.coefficients.iter().position(|c| !c.is_zero()).unwrap_or(self.coefficients.len())
    }
}

impl<T: Float> PiecewisePolynomial<T> {
    pub fn new(start: T, pieces: Vec<(T, Polynomial<T>)>, tail: Polynomial<T>) -> Self {
        Self { start, pieces, tail }
    }

    /// Highest polynomial degree over the pieces and the tail.
    pub fn degree(&self) -> usize {
        self.pieces
            .iter()
            .map(|(_, p)| p)
            .chain(std::iter::once(&self.tail))
            .map(|p| p.len().saturating_sub(1))
            .max()
            .unwrap_or(0)
    }

    /// The polynomial in effect at `t` and the local time `τ`; `None` before `start`.
    /// At a breakpoint the following piece is used.
    fn local(&self, t: T) -> Option<(&Polynomial<T>, T)> {
        if t < self.start {
            return None;
        }
        let mut begin = self.start;
        for (duration, p) in &self.pieces {
            if t < begin + *duration {
                return Some((p, t - begin));
            }
            begin = begin + *duration;
        }
        Some((&self.tail, t - begin))
    }

    /// `y(t)`.
    pub fn value(&self, t: T) -> T {
        self.derivatives(t, 1)[0]
    }

    /// `[y(t), y'(t), ..., y^(count-1)(t)]` (right-hand values at breakpoints).
    pub fn derivatives(&self, t: T, count: usize) -> Vec<T> {
        let Some((p, tau)) = self.local(t) else {
            return vec![T::zero(); count];
        };
        let mut coeffs = p.0.clone();
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(coeffs.iter().fold(T::zero(), |acc, &c| acc * tau + c));
            coeffs = derivative(&coeffs);
        }
        out
    }

    /// Breakpoints and their jumps `Δ_q` (see the module documentation).
    pub(super) fn jumps(&self) -> Vec<Jump<T>> {
        let mut jumps = Vec::with_capacity(self.pieces.len() + 1);
        let mut time = self.start;
        // (previous piece in ascending order, its duration)
        let mut previous: Option<(Vec<T>, T)> = None;
        for next in self.pieces.iter().map(|(d, p)| (*d, p)).chain(std::iter::once((T::zero(), &self.tail))) {
            let (duration, p) = next;
            let current = ascending(p);
            let coefficients = match &previous {
                None => current.clone(),
                Some((prev, length)) => {
                    let shifted = shift(prev, *length);
                    let len = current.len().max(shifted.len());
                    let at = |v: &[T], i: usize| v.get(i).copied().unwrap_or_else(T::zero);
                    // Natural size of the m-th coefficient: max_i |a_i| T^i / T^m.
                    let natural = |v: &[T]| {
                        v.iter().enumerate().fold(T::zero(), |acc, (i, &c)| acc.max(c.abs() * length.powi(i as i32)))
                    };
                    let scale = natural(&current).max(natural(prev));
                    let tol = T::from(JUMP_RELATIVE_TOLERANCE).unwrap();
                    (0..len)
                        .map(|m| {
                            let c = at(&current, m) - at(&shifted, m);
                            let threshold = tol * scale / length.powi(m as i32);
                            if c.abs() <= threshold { T::zero() } else { c }
                        })
                        .collect()
                }
            };
            let mut coefficients: Vec<T> = coefficients;
            while coefficients.last().is_some_and(|c| c.is_zero()) {
                coefficients.pop();
            }
            if !coefficients.is_empty() {
                jumps.push(Jump { time, coefficients });
            }
            previous = Some((current, duration));
            time = time + duration;
        }
        jumps
    }
}

impl<T: Float + std::ops::AddAssign> PiecewisePolynomial<T> {
    /// Exact Laplace transform `Σ_q e^(-s t_q) Σ_m c_m m! / s^(m+1)` (jump decomposition).
    /// Note that `LaplaceSignal::inverse_laplace` of it loses accuracy long after the
    /// breakpoints; use `value` / `derivatives` to evaluate `y`.
    pub fn laplace(&self) -> LaplaceSignal<T> {
        let mut signal = LaplaceSignal::new();
        for jump in self.jumps() {
            signal.push(jump.time, jump_rational(&jump));
        }
        signal
    }
}

/// `L[Σ c_m τ^m] = Σ c_m m! / s^(m+1) = (Σ c_m m! s^(M-1-m)) / s^M`, `M = len`.
pub(super) fn jump_rational<T: Float>(jump: &Jump<T>) -> TransferFunction<T, Continuous> {
    let len = jump.coefficients.len();
    let mut factorial = T::one();
    let mut numerator = vec![T::zero(); len];
    for (m, &c) in jump.coefficients.iter().enumerate() {
        if m > 0 {
            factorial = factorial * T::from(m).unwrap();
        }
        // descending index m <-> power s^(M-1-m)
        numerator[m] = c * factorial;
    }
    let mut denominator = vec![T::zero(); len + 1];
    denominator[0] = T::one();
    TransferFunction::from_polynomials(Polynomial(numerator), Polynomial(denominator))
}

fn ascending<T: Float>(p: &Polynomial<T>) -> Vec<T> {
    p.iter().rev().copied().collect()
}

/// Descending derivative.
fn derivative<T: Float>(coeffs: &[T]) -> Vec<T> {
    let degree = coeffs.len().saturating_sub(1);
    coeffs
        .iter()
        .take(degree)
        .enumerate()
        .map(|(i, &c)| c * T::from(degree - i).unwrap())
        .collect()
}

/// Ascending coefficients of `p(τ + a)` from ascending `p`: `b_m = Σ_{i>=m} C(i, m) a^(i-m) p_i`.
fn shift<T: Float>(p: &[T], a: T) -> Vec<T> {
    let n = p.len();
    (0..n)
        .map(|m| {
            let mut binomial = T::one(); // C(i, m) for i = m
            let mut power = T::one(); // a^(i - m)
            let mut sum = T::zero();
            for (i, &c) in p.iter().enumerate().skip(m) {
                if i > m {
                    binomial = binomial * T::from(i).unwrap() / T::from(i - m).unwrap();
                    power = power * a;
                }
                sum = sum + binomial * power * c;
            }
            sum
        })
        .collect()
}
