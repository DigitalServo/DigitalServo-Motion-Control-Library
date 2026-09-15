use std::borrow::Borrow;
use std::collections::VecDeque;
use std::ops::{AddAssign, MulAssign};
use num_traits::Float;

use crate::{Polynomial, TransferFunction};
use crate::math::binomial_coefficient;

/// Descending order of powers for (1 + x)^n
fn binom_one_plus_x<T: Float>(n: usize) -> Polynomial<T> {
    let mut poly = vec![T::zero(); n + 1];
    for k in 0..=n {
        poly[k] = T::from(binomial_coefficient(n, k)).unwrap();
    }
    Polynomial(poly)
}

/// Descending order of powers for (1 - x)^n
fn binom_one_minus_x<T: Float>(n: usize) -> Polynomial<T> {
    let mut poly: Polynomial<T> = binom_one_plus_x(n);
    for i in (1..=n).step_by(2) {
        poly[i] = -poly[i];
    }
    poly
}

/// Receive two descending-order polynomialnomial slices and bilinear transform them.
///
/// Numerator: bm*s^m + bm-1*s^m-1 + ...+ b0 => \[bm, bm-1, ..., b0\]
///
/// Denominator: an*s^n + an-1*s^n-1 + ...+ a0 => \[an, an-1, ..., a0\]
pub fn discretize<T, S>(
    tf: S,
    ts: T,
) -> TransferFunction<T>
where
    T: Float + AddAssign + MulAssign,
    S: Borrow<TransferFunction<T>>
{
    let tf = tf.borrow();

    let n = tf.denominator.len() - 1;
    let mut numer_z = Polynomial::zeros(n);
    let mut denom_z = Polynomial::zeros(n);

    let alpha = T::from(2.0).unwrap() / ts;

    // Numerator： sum of [(bk * α^k * (1 - q)^k) * (1 + q)^{N - k}]
    for (k, &bk) in tf.numerator.iter().rev().enumerate() {
        if bk != T::zero() {
            let scaler = bk * alpha.powi(k as i32);
            let term = &binom_one_minus_x(k) * &binom_one_plus_x(n - k);
            numer_z += &(term * scaler);
        }
    }

    // Denominator： sum of [(ak * α^k * (1 - q)^k) * (1 + q)^{N - k}]
    for (k, &ak) in tf.denominator.iter().rev().enumerate() {
        if ak != T::zero() {
            let scalar = ak * alpha.powi(k as i32);
            let term = &binom_one_minus_x(k) * &binom_one_plus_x(n - k);
            denom_z += &(&term * scalar);
        }
    }

    let scale = T::one() / denom_z[0];
    numer_z *= scale;
    denom_z *= scale;

    TransferFunction { numerator: numer_z, denominator: denom_z }
}

/// y\[k\] = (bn\[0\] * x\[k\] + bn\[1\] * x\[k-1\] + ... + bn\[N\] * x\[k-N\]) - (an\[0\] * y\[k-1\] + an\[1\] * y\[k-2\] - ... + an\[N\] * y\[k-N-1\])
pub struct DiscretizedSystem<T> {
    an: Vec<T>,
    bn: Vec<T>,
    xz: VecDeque<T>,
    yz: VecDeque<T>,
    pub output: T,
}

impl <T:Float + AddAssign + MulAssign> DiscretizedSystem<T> {
    pub fn new<S: Borrow<TransferFunction<T>>>(tf: S, ts: T) -> Self {

        let tf_z = discretize(tf, ts);
        let numer_order = tf_z.numerator.len() - 1;
        let denom_order = tf_z.denominator.len() - 1;
        let relative_order = denom_order - numer_order;

        let an = tf_z.denominator[1..].to_vec();
        let mut bn= vec![T::zero(); relative_order];
        bn.extend_from_slice(&tf_z.numerator);

        let xz = vec![T::zero(); denom_order + 1].into();
        let yz = vec![T::zero(); denom_order].into();

        Self { an, bn, xz, yz, output: T::zero() }
    }

    pub fn from_tf_z<S: Borrow<TransferFunction<T>>>(tf_z: S) -> Self {

        let tf_z = tf_z.borrow();
        let numer_order = tf_z.numerator.len() - 1;
        let denom_order = tf_z.denominator.len() - 1;
        let relative_order = denom_order - numer_order;

        let an = tf_z.denominator[1..].to_vec();
        let mut bn= vec![T::zero(); relative_order];
        bn.extend_from_slice(&tf_z.numerator);

        let xz = vec![T::zero(); denom_order + 1].into();
        let yz = vec![T::zero(); denom_order].into();

        Self { an, bn, xz, yz, output: T::zero() }
    }

    pub fn update(&mut self, x: T) -> T {
        // FIFO for xz
        self.xz.pop_back();
        self.xz.push_front(x);

        let yb = self.bn.iter().zip(self.xz.iter()).fold(T::zero(), |acc, (a, b)| acc + *a * *b);
        let ya = self.an.iter().zip(self.yz.iter()).fold(T::zero(), |acc, (a, b)| acc + *a * *b);
        self.output = yb - ya;

        // FIFO for yz
        self.yz.pop_back();
        self.yz.push_front(self.output);

        self.output
    }
}
