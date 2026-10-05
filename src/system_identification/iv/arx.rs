//! Instrumental-variable identification of an ARX model.

use std::ops::{AddAssign, MulAssign};
use nalgebra::{ComplexField, DMatrix, DVector};
use num_traits::Float;

use crate::{Discrete, TransferFunction};
use crate::system_identification::arx::Arx;
use crate::system_identification::lsm;

/// Instrumental-variable identification of the ARX model
/// `y[k] = Σ_{i=1..na} a_i y[k-i] + Σ_{i=0..nb} b_i u[k-nk-i]` from sequential data.
///
/// With measurement noise on the output, the regressor `φ[k]` (which contains past measured
/// outputs) is correlated with the equation error and least squares is biased. The IV method
/// replaces `φ` by instruments `ζ` correlated with `φ` but not with the noise:
///
/// ```text
/// θ = (Σ ζ[k] φ[k]ᵀ)^-1 Σ ζ[k] y[k]
/// ζ[k] = [x̂[k-1], ..., x̂[k-na], u[k-nk], ..., u[k-nk-nb]],  x̂[k] = ζ[k]ᵀ θ_aux
/// ```
///
/// where `x̂` is the noise-free output of an auxiliary model `θ_aux` driven by the input alone
/// (typically the least-squares estimate; see `identify`). The auxiliary model must be stable.
/// Call `add` every sample with `u[k]`, `y[k-1]` and `y[k]`, then `identify`.
pub struct DataBuffer<T> {
    /// Model; `identify` sets `arx.parameter`.
    pub arx: Arx<T>,
    /// Auxiliary model generating the instruments; its output history is the simulated `x̂`.
    pub auxiliary: Arx<T>,
    x_hat: T,
    zeta_y_sum: DVector<T>,
    zeta_phi_sum: DMatrix<T>,
}

impl<T: Float + AddAssign + MulAssign + ComplexField> DataBuffer<T> {
    /// `auxiliary`: model generating the instruments. Its orders and input delay are also those of
    /// the identified model; its history is cleared.
    pub fn new(auxiliary: &Arx<T>) -> Self {
        let n = auxiliary.parameter_len();
        let mut auxiliary = auxiliary.clone();
        auxiliary.clear_history();
        let mut arx = auxiliary.clone();
        arx.parameter.fill(T::zero());
        Self {
            arx,
            auxiliary,
            x_hat: T::zero(),
            zeta_y_sum: DVector::zeros(n),
            zeta_phi_sum: DMatrix::zeros(n, n),
        }
    }

    /// Add one sample: input `u = u[k]`, previous output `x = y[k-1]`, and output `y = y[k]`.
    pub fn add(&mut self, u: T, x: T, y: T) {
        self.arx.push(u, x);
        let phi = self.arx.regressor();

        self.auxiliary.push(u, self.x_hat);
        let zeta = self.auxiliary.regressor();
        self.x_hat = self.auxiliary.predict();

        self.zeta_y_sum += &zeta * y;
        self.zeta_phi_sum += &zeta * &phi.transpose();
    }

    /// Identify the parameters into `arx.parameter` and return `G(z)`; `None` (parameters unchanged)
    /// if `Σ ζ φᵀ` is singular.
    pub fn identify(&mut self) -> Option<TransferFunction<T, Discrete>> {
        self.arx.parameter = self.zeta_phi_sum.clone().lu().solve(&self.zeta_y_sum)?;
        Some(self.arx.transfer_function())
    }
}


/// Iterative IV identification from the input / output series with the orders and input delay of
/// `structure` (its parameters and history are not used): least squares first, then `iterations`
/// IV steps, each with the previous estimate as the auxiliary model. One iteration already removes
/// the bias of least squares; further ones reduce the variance. Returns the identified model
/// (`transfer_function()` gives `G(z)`), or `None` if a step is singular.
pub fn identify<T>(u: &[T], y: &[T], structure: &Arx<T>, iterations: usize) -> Option<Arx<T>>
where
    T: Float + AddAssign + MulAssign + ComplexField,
{
    let y_prev = |k: usize| if k > 0 { y[k - 1] } else { T::zero() };

    let mut arx = structure.clone();
    arx.clear_history();
    let mut buffer = lsm::arx::DataBuffer::from_arx(arx);
    for (k, (&uk, &yk)) in u.iter().zip(y).enumerate() {
        buffer.add(uk, y_prev(k), yk);
    }
    buffer.identify()?;
    let mut model = buffer.arx;

    for _ in 0..iterations {
        let mut buffer = DataBuffer::new(&model);
        for (k, (&uk, &yk)) in u.iter().zip(y).enumerate() {
            buffer.add(uk, y_prev(k), yk);
        }
        buffer.identify()?;
        model = buffer.arx;
    }
    Some(model)
}
