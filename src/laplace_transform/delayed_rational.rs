//! Signals given exactly in the Laplace domain as a sum of delayed rational functions,
//! `Y(s) = Σ_i e^(-s τ_i) R_i(s)`, i.e. `y(t) = Σ_i r_i(t - τ_i)` with causal `r_i`.
//! Piecewise trajectories (polynomial, sinusoidal, ...) are written this way by starting
//! each piece at its own delay.

use crate::{Continuous, TransferFunction};
use num_traits::Float;
use std::ops::AddAssign;

/// `e^(-s delay) rational(s)`.
#[derive(Clone, Debug)]
pub struct DelayedRational<T> {
    pub delay: T,
    pub rational: TransferFunction<T, Continuous>,
}

/// `Y(s) = Σ_i e^(-s τ_i) R_i(s)`.
#[derive(Clone, Debug)]
pub struct DelayedRationalSum<T> {
    pub terms: Vec<DelayedRational<T>>,
}

impl<T> DelayedRationalSum<T> {
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Add `e^(-s delay) rational(s)`.
    pub fn push(&mut self, delay: T, rational: TransferFunction<T, Continuous>) -> &mut Self {
        self.terms.push(DelayedRational { delay, rational });
        self
    }
}

impl<T> Default for DelayedRationalSum<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + AddAssign> DelayedRationalSum<T> {
    /// `y(t)` as a closure (causal inverse Laplace transform of each component, shifted by its delay).
    /// Impulse terms are not included, as in `PartialFraction::time_response`.
    pub fn inverse_laplace(&self) -> impl Fn(T) -> T + use<T>
    where
        T: 'static,
    {
        let parts: Vec<_> = self
            .terms
            .iter()
            .map(|c| (c.delay, c.rational.partial_fraction()))
            .collect();
        move |t| {
            parts
                .iter()
                .fold(T::zero(), |acc, (delay, pf)| acc + pf.time_response(t - *delay))
        }
    }
}
