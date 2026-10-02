//! Levy's method and the Sanathanan-Koerner iteration.

use std::ops::{AddAssign, MulAssign};

use nalgebra::{Complex, ComplexField, DMatrix, DVector};
use num_traits::{Float, Zero};
use thiserror::Error;

use crate::{Continuous, FrequencyResponse, Polynomial, TransferFunction};

/// Errors of Levy's method and the Sanathanan-Koerner iteration.
#[derive(Error, Debug)]
pub enum LevyIdentificationError {
    /// No samples.
    #[error("No sampled data provided")]
    EmptyData,
    /// The normal equations are singular.
    #[error("Matrix is singular or numerically unstable (check order and data)")]
    SingularMatrix,
    /// No iteration was run (`iterations == 0`).
    #[error("Failed to iterate (check data and iterations)")]
    IterationError,
    /// The previous estimate has a different denominator order.
    #[error("Order of previous result does not match current order")]
    OrderMismatch,
}

/// Levy's method (linear least squares on `B(jω) - G(jω) A(jω)` for `G(s) = B(s) / A(s)`):
/// `G(s)` with numerator degree `numer_order` and denominator degree `denom_order`.
/// `omega` of the samples is in rad/s.
pub fn identify<T: Float + ComplexField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    numer_order: usize,
    denom_order: usize,
) -> Result<TransferFunction<T, Continuous>, LevyIdentificationError> {
    levy_step(samples, numer_order, denom_order, None)
}

/// Levy's method followed by Sanathanan-Koerner reweighting, `iterations` steps in total
/// (`iterations = 1` is Levy's method).
pub fn sanathanan_koerner_identification<T: Float + ComplexField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    numer_order: usize,
    denom_order: usize,
    iterations: usize,
) -> Result<TransferFunction<T, Continuous>, LevyIdentificationError> {
    let mut ret = None;
    for _ in 0..iterations {
        ret = match levy_step(samples, numer_order, denom_order, ret) {
            Ok(tf) => Some(tf),
            Err(e) => return Err(e),
        };
    }

    match ret {
        Some(tf) => Ok(tf.clone()),
        None => Err(LevyIdentificationError::IterationError),
    }
}

fn levy_step<T: Float + ComplexField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    numer_order: usize,
    denom_order: usize,
    prev_result: Option<TransferFunction<T, Continuous>>,
) -> Result<TransferFunction<T, Continuous>, LevyIdentificationError> {
    let num_data = samples.len();
    if num_data == 0 {
        return Err(LevyIdentificationError::EmptyData);
    }

    if let Some(prev) = &prev_result {
        if prev.denominator.len() != (denom_order + 1) {
            return Err(LevyIdentificationError::OrderMismatch);
        }
    }

    let (num_a, num_b) = (denom_order, numer_order + 1);
    let k = num_a + num_b;

    let mut r = DMatrix::<T>::zeros(k, k);
    let mut rhs = DVector::<T>::zeros(k);

    let s0 = Complex::new(T::one(), T::zero());
    let powers_size = std::cmp::max(denom_order + 1, numer_order);

    for sample in samples {
        let w = sample.omega;
        let gr = sample.value.re;
        let gi = sample.value.im;

        let s = Complex::<T>::new(T::zero(), w);
        let mut powers = Vec::with_capacity(powers_size);
        let mut current = s0;
        powers.push(current);
        for _ in 1..powers_size {
            current *= s;
            powers.push(current);
        }
        let sn = powers[denom_order];

        let w_p = match &prev_result {
            Some(ret) => {
                let denom_coeffs = &ret.denominator.0;
                let mut denom = Complex::zero();
                for i in 0..(denom_order + 1) {
                    denom += Complex::from(denom_coeffs[denom_order - i]) * powers[i];
                }
                T::one() / denom.norm()
            },
            None => T::one()
        };

        let phi = {
            let mut ret = DMatrix::<T>::zeros(2, k);

            for j in 0..num_a {
                let p = powers[num_a - 1 - j];
                ret[(0, j)] = gr * p.re - gi * p.im;
                ret[(1, j)] = gr * p.im + gi * p.re;
            }

            for j in 0..num_b {
                let p = powers[num_b - 1 - j];
                ret[(0, j + num_a)] = -p.re;
                ret[(1, j + num_a)] = -p.im;
            }

            ret * w_p
        };

        let y = {
            let mut ret = DVector::<T>::zeros(2);
            ret[0] = gr * sn.re - gi * sn.im;
            ret[1] = gr * sn.im + gi * sn.re;
            ret * w_p
        };

        r += &phi.transpose() * &phi;
        rhs += &phi.transpose() * &y;

    }

    let theta = match r.qr().solve(&(-rhs)) {
        Some(x) => x.data.as_slice().to_owned(),
        None => return Err(LevyIdentificationError::SingularMatrix),
    };

    let a_coeffs_desc = &theta[0..num_a];
    let b_coeffs_desc = &theta[num_a..];

    let numer_coeffs = b_coeffs_desc.to_vec();
    let mut denom_coeffs = vec![T::one()];
    denom_coeffs.extend(a_coeffs_desc);

    Ok(TransferFunction::from_polynomials(Polynomial(numer_coeffs), Polynomial(denom_coeffs)))
}
