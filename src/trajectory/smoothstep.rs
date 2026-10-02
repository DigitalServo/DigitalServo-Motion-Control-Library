//! Smooth rest-to-rest polynomial trajectory of arbitrary smoothness (smoothstep).

use num_traits::Float;

use crate::laplace_transform::PiecewisePolynomial;
use crate::trajectory::{ReferenceTrajectory, Trajectory, TrajectoryProfile};
use crate::Polynomial;

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

/// A move of `distance` in `duration` \[s\] starting at `start` \[s\] (0 before, `distance` after),
/// `C^k` with `k = smoothness`. Use it as the reference of `ReferenceSignal::to_state_reference`,
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

/// Smooth rest-to-rest polynomial `distance P_k(x)`, `C^k` with `k = smoothness`, also called
/// "smoothstep" or "minimum-derivative trajectory":
///
/// ```text
/// P_k(x) = x^(k+1) Σ_{j=0..k} C(k+j, j) (1 - x)^j      (degree 2k + 1, 0 <= x <= 1)
/// ```
/// is the unique polynomial of degree `2k + 1` with `P(0) = 0`, `P(1) = 1` and derivatives
/// 1..k vanishing at both ends, so the move is `C^k` (the `(k+1)`-th derivative jumps).
/// `k = 1`: cubic, `k = 2`: quintic (minimum jerk), `k = 3`: septic, ...
///
/// # Smoothness for perfect tracking control
///
/// For a plant `N(s) / D(s)` of order `n` and relative degree `ρ = n - deg N`, the state reference
/// (`ReferenceSignal::to_state_reference`) contains `y_d` up to its `(ρ - 1)`-th derivative
/// (the zeros of `N` only filter it), so it is free of impulses iff `y_d^(ρ - 1)` is, i.e.
///
/// ```text
/// k >= ρ - 2
/// ```
///
/// (otherwise `StableInverseError::NotSmoothEnough`). E.g. `ρ = 1`: any `k`, `ρ = 2`: `k >= 0`,
/// `ρ = 3`: `k >= 1`. The input `u = D(d/dt) ξ_d` needs `y_d^(ρ)`, which is not required to be
/// bounded since the lifted PTC input only matches the state at frame instants; with the minimum
/// `k` the input becomes large pulses at both ends of the move (scaling with `1 / (n ts)`), so use
/// `k >= ρ - 1` (bounded continuous-time input) or larger in practice.
#[derive(Clone, Copy, Debug)]
pub struct SmoothPolynomial {
    /// Number of derivatives that vanish at both ends (`k`; degree `2k + 1`).
    pub smoothness: usize,
}

impl<T: Float> Trajectory<T> for SmoothPolynomial {
    fn profile(&self, distance: T, x: T) -> TrajectoryProfile<T> {
        if x < T::zero() {
            return TrajectoryProfile::rest(T::zero());
        }
        if x > T::one() {
            return TrajectoryProfile::rest(distance);
        }
        // Horner on ascending coefficients for P, P', P''
        let coeffs = normalized_coefficients::<T>(self.smoothness);
        let (mut p, mut dp, mut ddp) = (T::zero(), T::zero(), T::zero());
        for &c in coeffs.iter().rev() {
            ddp = ddp * x + dp + dp;
            dp = dp * x + p;
            p = p * x + c;
        }
        TrajectoryProfile { s: distance * p, v: distance * dp, a: distance * ddp }
    }
}

impl<T: Float> ReferenceTrajectory<T> for SmoothPolynomial {
    type Reference = PiecewisePolynomial<T>;

    /// Exact piecewise-polynomial form (see `piecewise`).
    fn reference(&self, distance: T, duration: T, start: T) -> PiecewisePolynomial<T> {
        piecewise(distance, duration, start, self.smoothness)
    }
}
