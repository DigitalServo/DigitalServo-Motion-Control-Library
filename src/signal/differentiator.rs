use nalgebra::{RealField, DMatrix, DVector, Scalar};
use num_traits::Float;

use crate::binomial_coefficients;

/// Pseudo-differentiator based on a minimal-order state observer of a chain of integrators,
/// with all observer poles at `-bandwidth`. `filter_order` adds extra low-pass states.
/// `update` returns the `derivative_order`-th derivative; `output[i]` holds the `(i + 1)`-th one.
#[derive(Debug, Clone)]
pub struct Differentiator<T>
{
    ts: T,
    g: DVector<T>,
    ty: DVector<T>,
    tz: DMatrix<T>,
    py: DVector<T>,
    py_z1: DVector<T>,
    derivative_order: usize,
    /// Estimated derivatives of the last `update`: `output[i]` is the `(i + 1)`-th derivative.
    pub output: Vec<T>
}

impl<T: Float + RealField> Differentiator<T> {
    /// `ts`: sampling period, `bandwidth`: observer pole magnitude \[rad/s\], `derivative_order`: derivative
    /// returned by `update` (at least 1), `filter_order`: extra low-pass order.
    pub fn new(ts: T, bandwidth: T, derivative_order: usize, filter_order: usize) -> Self {

        let state_order = derivative_order + filter_order;

        /* gain vector which realizes multiple root */
        let pascal_coeff: Vec<T> = binomial_coefficients(state_order)
            .into_iter()
            .map(|x| T::from(x).unwrap())
            .collect();
        let mut g: DVector<T> = DVector::zeros(state_order);
        for i in 0..state_order {
            g[i] = pascal_coeff[i + 1] * <T as Float>::powi(bandwidth, i as i32 + 1);
        }

        /* minimum-order state observer */
        /* see https://digitalservo.jp/library/linear-control-design/observer-design/minimal-order-observer/ */

        //system matrix
        let a_11: DMatrix<T> = jordan_block(T::zero(), state_order);
        let a_12: DVector<T> = DVector::zeros(state_order);
        let mut a_21: DVector<T> = DVector::zeros(state_order);
        a_21[0] = T::one();
        let a_22: T = T::zero();

        //matrices for state observer
        let ty: DVector<T> = &a_12 - &g * a_22;
        let tz: DMatrix<T> = &a_11 - &g * a_21.transpose();

        //initialize
        let py: DVector<T> = DVector::zeros(state_order);
        let py_z1: DVector<T> = DVector::zeros(state_order);

        let output: Vec<T> = vec![T::zero(); state_order];

        Self {ts, g, ty, tz, py, py_z1, output, derivative_order}
    }

    /// One sample of the signal `x`; returns its `derivative_order`-th derivative.
    pub fn update(&mut self, x: T) -> T {
        let u: DVector<T> = (&self.tz * &self.g + &self.ty) * x;
        self.py += (u + &self.tz * &self.py) * self.ts;
        self.output = (&self.py_z1 + (&self.g * x)).data.as_vec().to_vec();
        self.py_z1 = self.py.clone();
        self.output[self.derivative_order - 1]
    }
}

fn jordan_block<T: Float + Scalar>(lambda: T, order: usize) -> DMatrix<T> {
    DMatrix::from_fn(order, order, |i, j| {
        match (i, j) {
            (i, j) if i == j => lambda,
            (i, j) if j == i + 1 => T::one(),
            _ => T::zero(),
        }
    })
}
