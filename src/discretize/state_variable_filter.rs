//! Exact discretization of the state-variable filter `1 / A(s)` for a held sampled signal, scaled
//! so that it stays accurate at high orders. Used for continuous-time simulation
//! (`TransferFunctionWithDelay::simulate`) and by SRIVC (its prefilter).

use std::ops::AddAssign;

use nalgebra::{ComplexField, DMatrix, DVector, RealField};
use num_traits::Float;

/// Intersample behaviour of a sampled signal, used to filter it in continuous time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterSample {
    /// Constant between samples (an input applied through a D/A converter); exact for such inputs.
    ZeroOrderHold,
    /// Linear between samples (a smooth signal such as a measured output).
    FirstOrderHold,
}

/// Filter `1 / A(s)` of a sampled signal `v`, giving `[x, x', ..., x^(n)]` of `x = v / A(s)` at the
/// sampling instants (zero initial state).
///
/// The coefficients `a_i` grow like `ρ^i` (`ρ`: radius of the roots), so the companion matrix in
/// the states `x^(i)` spans many decades at high orders and its exponential is inaccurate. The
/// states are therefore scaled as in the normalized time `ρ t`: `z_i = x^(i) / ρ^i`, with the
/// input `w = v / ρ^n`, `ż = ρ (A' z + B w)`, `A'` the companion matrix of `a_i / ρ^i`.
pub(crate) struct StateVariableFilter<T> {
    a: Vec<T>,
    rho: T,
    phi: DMatrix<T>,
    gamma0: DVector<T>,
    gamma1: DVector<T>,
}

impl<T: Float + AddAssign + ComplexField + RealField> StateVariableFilter<T> {
    /// `a = [a_1, ..., a_n]` of the monic `A(s)`. Over a sampling period with the input
    /// `w(t_k + τ) = w[k] + (w[k+1] - w[k]) τ / ts`,
    /// `z[k+1] = Φ z[k] + Γ0 w[k] + Γ1 (w[k+1] - w[k])`: from the exponential of the augmented
    /// matrix `[[ρ ts A', ρ ts B, 0], [0, 0, 1], [0, 0, 0]] = [[Φ, Γ0, Γ1], ...]`.
    pub(crate) fn new(a: &[T], ts: T) -> Self {
        let n = a.len();
        let rho = root_radius(a);
        let rho_ts = rho * ts;
        let mut aug = DMatrix::<T>::zeros(n + 2, n + 2);
        for i in 0..n - 1 {
            aug[(i, i + 1)] = rho_ts;
        }
        for j in 0..n {
            aug[(n - 1, j)] = -a[n - 1 - j] / Float::powi(rho, (n - j) as i32) * rho_ts;
        }
        aug[(n - 1, n)] = rho_ts;
        aug[(n, n + 1)] = T::one();
        let e = aug.exp();
        Self {
            a: a.to_vec(),
            rho,
            phi: e.view((0, 0), (n, n)).into_owned(),
            gamma0: e.view((0, n), (n, 1)).column(0).into_owned(),
            gamma1: e.view((0, n + 1), (n, 1)).column(0).into_owned(),
        }
    }

    /// Order `n` of `A(s)`.
    pub(crate) fn order(&self) -> usize {
        self.a.len()
    }

    /// Row `k`: `[x[k], x'[k], ..., x^(n)[k]]`.
    pub(crate) fn apply(&self, v: &[T], hold: InterSample) -> DMatrix<T> {
        let n = self.a.len();
        let scale: Vec<T> = (0..=n).map(|i| Float::powi(self.rho, i as i32)).collect();
        let mut out = DMatrix::zeros(v.len(), n + 1);
        let mut z = DVector::zeros(n);
        for k in 0..v.len() {
            // x^(n) = v - a_1 x^(n-1) - ... - a_n x
            let mut highest = v[k];
            for i in 0..n {
                out[(k, i)] = z[i] * scale[i];
                highest -= self.a[n - 1 - i] * out[(k, i)];
            }
            out[(k, n)] = highest;

            let slope = match hold {
                InterSample::FirstOrderHold if k + 1 < v.len() => v[k + 1] - v[k],
                _ => T::zero(),
            };
            z = &self.phi * &z + &self.gamma0 * (v[k] / scale[n]) + &self.gamma1 * (slope / scale[n]);
        }
        out
    }
}

/// Radius of the roots of the monic `s^n + a_1 s^(n-1) + ... + a_n`, `max_i |a_i|^(1/i)` (1 if all
/// `a_i` are zero).
pub(crate) fn root_radius<T: Float>(a: &[T]) -> T {
    let rho = a.iter().enumerate().fold(T::zero(), |acc, (i, &c)| acc.max(c.abs().powf(T::one() / T::from(i + 1).unwrap())));
    if rho > T::zero() { rho } else { T::one() }
}
