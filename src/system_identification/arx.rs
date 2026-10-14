//! ARX model structure shared by the identification methods.

use std::ops::AddAssign;

use nalgebra::{DVector, Scalar};
use num_complex::Complex;
use num_traits::Float;

use crate::{Discrete, Polynomial, TransferFunction, dka_method, vieta_formula};

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

    /// Whether the roots of the denominator `1 - a_1 z^-1 - ... - a_na z^-na` are inside the unit
    /// circle (Schur-Cohn step-down: every reflection coefficient `|k| < 1`).
    pub fn is_stable(&self) -> bool {
        let na = self.state_order;
        let mut c: Vec<T> = std::iter::once(T::one()).chain(self.parameter.iter().take(na).map(|&a| -a)).collect();
        if c.iter().any(|v| !v.is_finite()) {
            return false;
        }
        while c.len() > 1 {
            let n = c.len() - 1;
            let k = c[n] / c[0];
            if k.abs() >= T::one() {
                return false;
            }
            c = (0..n).map(|i| (c[i] - k * c[n - i]) / (T::one() - k * k)).collect();
        }
        true
    }

    /// The model with the roots `r` of the denominator outside the unit circle reflected to
    /// `1 / r̄` (same history and `b_i`; the gain changes, the shape of `|G(e^jω)|` does not).
    pub fn stabilized(&self) -> Self {
        let na = self.state_order;
        if self.is_stable() {
            return self.clone();
        }
        let c: Vec<Complex<T>> =
            std::iter::once(T::one()).chain(self.parameter.iter().take(na).map(|&a| -a)).map(|v| Complex::new(v, T::zero())).collect();
        let Some(roots) = dka_method(&Polynomial(c)) else {
            return self.clone();
        };
        let reflected: Vec<Complex<T>> = roots.iter().map(|r| if r.norm() > T::one() { r.conj().inv() } else { *r }).collect();
        let mut model = self.clone();
        for (a, c) in model.parameter.iter_mut().zip(vieta_formula(&reflected).0.iter().skip(1)) {
            *a = -c.re;
        }
        model
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

#[cfg(test)]
mod tests {
    use super::*;

    fn model(a: &[f64]) -> Arx<f64> {
        let mut m = Arx::<f64>::new(a.len(), 0);
        m.parameter.rows_mut(0, a.len()).copy_from_slice(a);
        m.parameter[a.len()] = 1.0;
        m
    }

    #[test]
    fn schur_cohn_stability() {
        // (1 - 0.9 z^-1)(1 - 0.8 z^-1), complex pair of radius √0.7, z = 1.1, z = -1, poles 1.1 and 0.8
        assert!(model(&[1.7, -0.72]).is_stable());
        assert!(model(&[1.5, -0.7]).is_stable());
        assert!(!model(&[1.1]).is_stable());
        assert!(!model(&[-1.0]).is_stable());
        assert!(!model(&[1.9, -0.88]).is_stable());
        assert!(model(&[]).is_stable());
    }

    #[test]
    fn stabilized_reflects_unstable_roots() {
        // Poles 1.25 and 0.8 -> 0.8 (double): 1 - 1.6 z^-1 + 0.64 z^-2
        let m = model(&[2.05, -1.0]).stabilized();
        assert!(m.is_stable());
        assert!((m.parameter[0] - 1.6).abs() < 1e-12 && (m.parameter[1] + 0.64).abs() < 1e-12, "{:?}", m.parameter);
        assert_eq!(m.parameter[2], 1.0);
    }
}
