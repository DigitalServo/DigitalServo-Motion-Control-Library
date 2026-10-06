//! ARX model structure shared by the identification methods.

use std::ops::AddAssign;

use nalgebra::{DVector, Scalar};
use num_traits::Float;

use crate::{Discrete, TransferFunction};

/// ARX model `y[k] = Σ_{i=1..na} a_i y[k-i] + Σ_{i=0..nb} b_i u[k-nk-i]`
/// (`na = state_order`, `nb = input_order`, `nk = input_delay`, 0 unless set by `with_input_delay`):
/// the orders, the parameters `θ = [a_1, ..., a_na, b_0, ..., b_nb]`, and the input / output
/// history giving the regressor `φ[k] = [y[k-1], ..., y[k-na], u[k-nk], ..., u[k-nk-nb]]`,
/// so that `y[k] = φ[k]ᵀ θ`.
#[derive(Clone, Debug)]
pub struct Arx<T> {
    /// Input history `u[k], u[k-1], ..., u[k-nk-nb]`.
    pub u: DVector<T>,
    /// Output history `y[k-1], ..., y[k-na]`.
    pub x: DVector<T>,
    /// Parameters `θ = [a_1, ..., a_na, b_0, ..., b_nb]` (zero until identified).
    pub parameter: DVector<T>,
    input_order: usize,
    state_order: usize,
    input_delay: usize,
}

impl<T: Float + AddAssign + Scalar> Arx<T> {
    /// `state_order`: `na`, `input_order`: `nb`.
    ///
    /// Argument order: the output (denominator) order `na` first, then the input (numerator)
    /// order `nb`, as in the notation `(na, nb, nk)`; e.g. the model
    /// `y[k] = a_1 y[k-1] + a_2 y[k-2] + b_0 u[k] + b_1 u[k-1]` is `(2, 1)`.
    pub fn new(state_order: usize, input_order: usize) -> Self {
        Self {
            u: DVector::zeros(input_order + 1),
            x: DVector::zeros(state_order),
            parameter: DVector::zeros(input_order + state_order + 1),
            input_order,
            state_order,
            input_delay: 0,
        }
    }

    /// Input delay `nk` \[samples\]: the model uses `u[k-nk] .. u[k-nk-nb]`. Clears the input history.
    pub fn with_input_delay(mut self, input_delay: usize) -> Self {
        self.input_delay = input_delay;
        self.u = DVector::zeros(input_delay + self.input_order + 1);
        self
    }

    /// `nb`.
    pub fn input_order(&self) -> usize {
        self.input_order
    }

    /// `na`.
    pub fn state_order(&self) -> usize {
        self.state_order
    }

    /// `nk`.
    pub fn input_delay(&self) -> usize {
        self.input_delay
    }

    /// Number of parameters `na + nb + 1`.
    pub fn parameter_len(&self) -> usize {
        self.state_order + self.input_order + 1
    }

    /// Clear the input / output history (the parameters are kept).
    pub fn clear_history(&mut self) {
        self.u.fill(T::zero());
        self.x.fill(T::zero());
    }

    /// Shift in input `u = u[k]` and previous output `x = y[k-1]`.
    pub fn push(&mut self, u: T, x: T) {
        //FIFO for input u
        for i in (1..self.u.len()).rev() {
            self.u[i] = self.u[i - 1]
        }
        self.u[0] = u;

        //FIFO for state x
        for i in (1..self.state_order).rev() {
            self.x[i] = self.x[i - 1]
        }
        if self.state_order > 0 {
            self.x[0] = x;
        }
    }

    /// Regressor `φ[k] = [y[k-1], ..., y[k-na], u[k-nk], ..., u[k-nk-nb]]` of the current history.
    pub fn regressor(&self) -> DVector<T> {
        let mut phi = DVector::zeros(self.parameter_len());
        for i in 0..self.state_order {
            phi[i] = self.x[i]
        }
        for i in 0..(self.input_order + 1) {
            phi[i + self.state_order] = self.u[i + self.input_delay]
        }
        phi
    }

    /// Output `φ[k]ᵀ θ` of the model for the current history.
    pub fn predict(&self) -> T {
        self.regressor().iter().zip(self.parameter.iter()).fold(T::zero(), |acc, (&p, &t)| acc + p * t)
    }

    /// `G(z)` of the parameters `θ`, i.e.
    ///
    /// ```text
    /// G(z) = z^-nk (b_0 + b_1 z^-1 + ... + b_nb z^-nb) / (1 - a_1 z^-1 - ... - a_na z^-na)
    /// ```
    ///
    /// as polynomials in `z` (both multiplied by `z^max(na, nk + nb)`), so that a lower-order side
    /// gets its `z` factors when the orders differ. Common poles / zeros are cancelled.
    pub fn transfer_function(&self) -> TransferFunction<T, Discrete> {
        let (a, b) = self.parameter.as_slice().split_at(self.state_order);
        let order = a.len().max(self.input_delay + b.len().saturating_sub(1));

        let mut denom = Vec::with_capacity(order + 1);
        denom.push(T::one());
        denom.extend(a.iter().map(|&ai| -ai));
        denom.resize(order + 1, T::zero());

        // z^-nk: nk leading zero coefficients of b in powers of z^-1
        let mut numer = vec![T::zero(); self.input_delay];
        numer.extend_from_slice(b);
        numer.resize(order + 1, T::zero());

        TransferFunction::discrete(&numer, &denom)
    }
}
