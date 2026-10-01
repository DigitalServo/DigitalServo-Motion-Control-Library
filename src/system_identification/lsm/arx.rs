use std::ops::{AddAssign, MulAssign};
use nalgebra::{ComplexField, DMatrix, DVector};
use num_traits::Float;

use crate::{Discrete, TransferFunction};

/// Least-squares identification of the ARX model
/// `y[k] = Σ_{i=1..na} a_i y[k-i] + Σ_{i=0..nb} b_i u[k-nk-i]` from sequential data
/// (`na = state_order`, `nb = input_order`, `nk = input_delay`, 0 unless set by `with_input_delay`).
pub struct DataBuffer<T> {
    /// Input history `u[k], u[k-1], ..., u[k-nk-nb]`.
    pub u: DVector<T>,
    pub x: DVector<T>,
    psi_sum: DVector<T>,
    phi_sum: DMatrix<T>,
    y2_sum: T,
    count: usize,
    input_order: usize,
    state_order: usize,
    input_delay: usize,
}

impl<T: Float + AddAssign + MulAssign + ComplexField> DataBuffer<T> {
    pub fn new(input_order: usize, state_order: usize) -> Self {
        Self {
            u: DVector::zeros(input_order + 1),
            x: DVector::zeros(state_order),
            psi_sum: DVector::zeros(input_order + state_order + 1),
            phi_sum: DMatrix::zeros(input_order + state_order + 1, input_order + state_order + 1),
            y2_sum: T::zero(),
            count: 0,
            input_order,
            state_order,
            input_delay: 0,
        }
    }

    /// Input delay `nk` [samples]: the model uses `u[k-nk] .. u[k-nk-nb]`, and `identify` gives
    /// `z^-nk B(z) / A(z)`. Set it before adding data.
    pub fn with_input_delay(mut self, input_delay: usize) -> Self {
        self.input_delay = input_delay;
        self.u = DVector::zeros(input_delay + self.input_order + 1);
        self
    }

    pub fn add(&mut self, u: T, x: T, y: T) {
        //FIFO for input u
        for i in (1..self.u.len()).rev() {
            self.u[i] = self.u[i - 1]
        }
        self.u[0] = u;

        //FIFO for state x
        for i in (1..self.state_order).rev() {
            self.x[i] = self.x[i - 1]
        }
        self.x[0] = x;

        let mut phi = DVector::zeros(self.input_order + self.state_order + 1);
        for i in 0..self.state_order {
            phi[i] = self.x[i]
        }
        for i in 0..(self.input_order + 1) {
            phi[i + self.state_order] = self.u[i + self.input_delay]
        }

        self.psi_sum += &phi * y;
        self.phi_sum += &phi * &phi.transpose();
        self.y2_sum += y * y;
        self.count += 1;
    }

    fn parameters(&self) -> Option<DVector<T>> {
        self.phi_sum.clone().try_inverse().map(|inv| &inv * &self.psi_sum)
    }

    /// Mean squared one-step prediction error `Σ (y - φᵀθ)^2 / N = (Σ y^2 - θᵀ Σ φ y) / N`.
    pub fn loss(&self) -> Option<T> {
        let theta = self.parameters()?;
        let count = T::from(self.count.max(1)).unwrap();
        Some(((self.y2_sum - theta.dot(&self.psi_sum)) / count).max(T::zero()))
    }

    pub fn identify(&self) -> Option<TransferFunction<T, Discrete>> {
        let theta = self.parameters()?;
        let (a, b) = theta.as_slice().split_at(self.state_order);
        Some(crate::system_identification::arx::transfer_function(a, b, self.input_delay))
    }
}

/// Result of `estimate_input_delay`.
#[derive(Clone, Debug)]
pub struct DelayEstimate<T> {
    /// Estimated input delay `nk` [samples].
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
pub fn estimate_input_delay<T>(
    u: &[T],
    y: &[T],
    input_order: usize,
    state_order: usize,
    max_delay: usize,
) -> Option<DelayEstimate<T>>
where
    T: Float + AddAssign + MulAssign + ComplexField,
{
    let buffers: Vec<DataBuffer<T>> = (0..=max_delay)
        .map(|nk| {
            let mut buffer = DataBuffer::new(input_order, state_order).with_input_delay(nk);
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
