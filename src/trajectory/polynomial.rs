//! Smooth rest-to-rest polynomial trajectory of arbitrary smoothness.
//!
//! ```text
//! P_k(x) = x^(k+1) Σ_{j=0..k} C(k+j, j) (1 - x)^j      (degree 2k + 1, 0 <= x <= 1)
//! ```
//! is the unique polynomial of degree `2k + 1` with `P(0) = 0`, `P(1) = 1` and derivatives
//! 1..k vanishing at both ends, so the move is `C^k` (the `(k+1)`-th derivative jumps).
//! `k = 1`: cubic, `k = 2`: quintic (minimum jerk), `k = 3`: septic, ...
//! With stable inversion, a plant of relative degree `ρ` needs `k >= ρ - 2`.

use std::ops::AddAssign;

use num_traits::Float;

use crate::trajectory::TrajectoryProfile;
use crate::{PiecewisePolynomial, Polynomial};

/// Ascending coefficients of `P_k(x)`.
pub fn normalized_coefficients<T: Float>(k: usize) -> Vec<T> {
    let mut coeffs = vec![T::zero(); 2 * k + 2];
    // C(k+j, j) (1 - x)^j = C(k+j, j) Σ_i C(j, i) (-x)^i, shifted by x^(k+1)
    let mut c_kj = T::one(); // C(k+j, j)
    for j in 0..=k {
        if j > 0 {
            c_kj = c_kj * T::from(k + j).unwrap() / T::from(j).unwrap();
        }
        let mut c_ji = T::one(); // C(j, i)
        for i in 0..=j {
            if i > 0 {
                c_ji = c_ji * T::from(j - i + 1).unwrap() / T::from(i).unwrap();
            }
            let sign = if i % 2 == 0 { T::one() } else { -T::one() };
            coeffs[k + 1 + i] = coeffs[k + 1 + i] + sign * c_kj * c_ji;
        }
    }
    coeffs
}

/// A move of `distance` in `duration` [s] starting at `start` [s] (0 before, `distance` after),
/// `C^k` with `k = smoothness`. Use it as the reference of `TransferFunction::state_reference`,
/// or evaluate it with `value` / `derivatives`.
pub fn piecewise<T: Float>(distance: T, duration: T, start: T, smoothness: usize) -> PiecewisePolynomial<T> {
    // p(τ) = distance P(τ / duration): coefficient of τ^i is distance P_i / duration^i
    let mut scale = distance;
    let descending: Vec<T> = normalized_coefficients::<T>(smoothness)
        .into_iter()
        .map(|c| {
            let v = c * scale;
            scale = scale / duration;
            v
        })
        .collect::<Vec<T>>()
        .into_iter()
        .rev()
        .collect();
    PiecewisePolynomial::new(start, vec![(duration, Polynomial(descending))], Polynomial(vec![distance]))
}

/// Samples like `sin::generate`: normalized time `x = 0..1` over `samples` points, with `v` and `a`
/// the derivatives with respect to `x` (divide by `duration` / `duration^2` for physical units).
/// `k = smoothness` means that the trajectory is `C^k` smooth.
pub fn generate<T: Float + AddAssign>(distance: T, samples: usize, smoothness: usize) -> Vec<TrajectoryProfile<T>> {
    let p = piecewise(distance, T::one(), T::zero(), smoothness);
    let dx = T::one() / T::from(samples - 1).unwrap();
    (0..samples)
        .map(|i| {
            // Evaluate the last sample on the piece itself (x = 1), not on the tail.
            let x = (T::from(i).unwrap() * dx).min(T::one() - T::epsilon());
            let d = p.derivatives(x, 3);
            TrajectoryProfile { s: d[0], v: d[1], a: d[2] }
        })
        .collect()
}
