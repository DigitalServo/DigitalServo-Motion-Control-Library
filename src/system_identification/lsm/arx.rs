//! Least-squares identification of an ARX model.

use std::ops::{AddAssign, MulAssign};
use nalgebra::{ComplexField, DMatrix, DVector};
use num_traits::Float;

use crate::{Discrete, TransferFunction};
use crate::system_identification::arx::Arx;

/// Least-squares identification of the ARX model
/// `y[k] = Σ_{i=1..na} a_i y[k-i] + Σ_{i=0..nb} b_i u[k-nk-i]` from sequential data
/// (`na = state_order`, `nb = input_order`, `nk = input_delay`, 0 unless set by `with_input_delay`).
/// Call `add` every sample with `u[k]`, `y[k-1]` and `y[k]`, then `identify`.
pub struct DataBuffer<T> {
    /// Model; `identify` sets `arx.parameter`.
    pub arx: Arx<T>,
    psi_sum: DVector<T>,
    phi_sum: DMatrix<T>,
    y2_sum: T,
    count: usize,
}

impl<T: Float + AddAssign + MulAssign + ComplexField> DataBuffer<T> {
    /// `state_order`: `na`, `input_order`: `nb`.
    ///
    /// Argument order: the output (denominator) order `na` first, then the input (numerator)
    /// order `nb`, as in the notation `(na, nb, nk)`; e.g. the model
    /// `y[k] = a_1 y[k-1] + a_2 y[k-2] + b_0 u[k] + b_1 u[k-1]` is `(2, 1)`.
    pub fn new(state_order: usize, input_order: usize) -> Self {
        Self::from_arx(Arx::new(state_order, input_order))
    }

    /// Least squares of the given model structure.
    pub fn from_arx(arx: Arx<T>) -> Self {
        let n = arx.parameter_len();
        Self {
            arx,
            psi_sum: DVector::zeros(n),
            phi_sum: DMatrix::zeros(n, n),
            y2_sum: T::zero(),
            count: 0,
        }
    }

    /// Input delay `nk` \[samples\]: the model uses `u[k-nk] .. u[k-nk-nb]`, and `identify` gives
    /// `z^-nk B(z) / A(z)`. Set it before adding data.
    pub fn with_input_delay(mut self, input_delay: usize) -> Self {
        self.arx = self.arx.with_input_delay(input_delay);
        self
    }

    /// Add one sample: input `u = u[k]`, previous output `x = y[k-1]`, and output `y = y[k]`.
    pub fn add(&mut self, u: T, x: T, y: T) {
        self.arx.push(u, x);
        let phi = self.arx.regressor();

        self.psi_sum += &phi * y;
        self.phi_sum += &phi * &phi.transpose();
        self.y2_sum += y * y;
        self.count += 1;
    }

    fn solve(&self) -> Option<DVector<T>> {
        self.phi_sum.clone().try_inverse().map(|inv| &inv * &self.psi_sum)
    }

    /// Mean squared one-step prediction error `Σ (y - φᵀθ)^2 / N = (Σ y^2 - θᵀ Σ φ y) / N`.
    pub fn loss(&self) -> Option<T> {
        let theta = self.solve()?;
        let count = T::from(self.count.max(1)).unwrap();
        Some(((self.y2_sum - theta.dot(&self.psi_sum)) / count).max(T::zero()))
    }

    /// Identify the parameters into `arx.parameter` and return `G(z)`; `None` (parameters unchanged)
    /// if the data do not determine them (singular normal equations).
    pub fn identify(&mut self) -> Option<TransferFunction<T, Discrete>> {
        self.arx.parameter = self.solve()?;
        Some(self.arx.transfer_function())
    }
}


/// Result of `estimate_input_delay`.
#[derive(Clone, Debug)]
pub struct DelayEstimate<T> {
    /// Estimated input delay `nk` \[samples\].
    pub input_delay: usize,
    /// ARX model identified with that delay, `z^-nk B(z) / A(z)`.
    pub model: TransferFunction<T, Discrete>,
    /// Mean squared one-step prediction error for `nk = 0 ..= max_delay` (`None` if singular).
    pub losses: Vec<Option<T>>,
}

/// Estimate the input delay of an ARX model of the given orders from the input / output series:
/// the model is identified for every `nk = 0 ..= max_delay` and the one with the smallest one-step
/// prediction error is chosen. With `input_order` larger than needed, smaller delays fit equally
/// well (their leading `b_i` are just zero), so among losses within `1e-9 mean(y^2)` of the minimum
/// the largest delay is taken. Returns `None` if no delay gives a non-singular problem.
///
/// Argument order: `state_order` (`na`) before `input_order` (`nb`), as in `(na, nb, nk)`.
pub fn estimate_input_delay<T>(
    u: &[T],
    y: &[T],
    state_order: usize,
    input_order: usize,
    max_delay: usize,
) -> Option<DelayEstimate<T>>
where
    T: Float + AddAssign + MulAssign + ComplexField,
{
    let mut buffers: Vec<DataBuffer<T>> = (0..=max_delay)
        .map(|nk| {
            let mut buffer = DataBuffer::new(state_order, input_order).with_input_delay(nk);
            for (k, (&uk, &yk)) in u.iter().zip(y).enumerate() {
                let y_prev = if k > 0 { y[k - 1] } else { T::zero() };
                buffer.add(uk, y_prev, yk);
            }
            buffer
        })
        .collect();
    let losses: Vec<Option<T>> = buffers.iter().map(|b| b.loss()).collect();

    let min = losses.iter().flatten().copied().reduce(T::min)?;
    let mean_y2 = y.iter().fold(T::zero(), |acc, &v| acc + v * v) / T::from(y.len().max(1)).unwrap();
    let threshold = min + T::from(1e-9).unwrap() * mean_y2;
    let input_delay = losses.iter().rposition(|l| l.is_some_and(|l| l <= threshold))?;
    let model = buffers[input_delay].identify()?;
    Some(DelayEstimate { input_delay, model, losses })
}
