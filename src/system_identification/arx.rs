//! Transfer function of an ARX model.

use std::ops::AddAssign;

use num_traits::Float;

use crate::{Discrete, TransferFunction};

/// `G(z)` of the ARX model `y[k] = Σ_{i=1..na} a_i y[k-i] + Σ_{i=0..nb} b_i u[k-nk-i]`, i.e.
///
/// ```text
/// G(z) = z^-nk (b_0 + b_1 z^-1 + ... + b_nb z^-nb) / (1 - a_1 z^-1 - ... - a_na z^-na)
/// ```
///
/// as polynomials in `z` (both multiplied by `z^max(na, nk + nb)`), so that a lower-order side gets
/// its `z` factors when the orders differ. Common poles / zeros are cancelled.
pub(crate) fn transfer_function<T: Float + AddAssign>(a: &[T], b: &[T], input_delay: usize) -> TransferFunction<T, Discrete> {
    let order = a.len().max(input_delay + b.len().saturating_sub(1));

    let mut denom = Vec::with_capacity(order + 1);
    denom.push(T::one());
    denom.extend(a.iter().map(|&ai| -ai));
    denom.resize(order + 1, T::zero());

    // z^-nk: nk leading zero coefficients of b in powers of z^-1
    let mut numer = vec![T::zero(); input_delay];
    numer.extend_from_slice(b);
    numer.resize(order + 1, T::zero());

    TransferFunction::discrete(&numer, &denom)
}
