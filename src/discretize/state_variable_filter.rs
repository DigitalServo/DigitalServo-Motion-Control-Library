//! Exact discretization of the state-variable filter `1 / A(s)` for a sampled signal (held or
//! linear between samples), scaled so that it stays accurate at high orders.
//!
//! Used for continuous-time simulation (`TransferFunctionWithDelay::simulate`), by SRIVC (its
//! filters and prefilter), and for continuous-time identification with a state-variable filter:
//! the derivatives `x^(i)` of `x = v / A(s)` are the pseudo-derivatives `s^i / A(s) v` of the
//! data, e.g. with `A(s) = (s + λ)^N` (`StateVariableFilter::lag`) the regressors of a
//! differential equation filtered by `F(s) = (λ / (s + λ))^N`.

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
/// The filter is discretized exactly for the intersample behaviour of `v` (`InterSample`): a
/// command output through a D/A converter is `ZeroOrderHold` (then exact), a measured continuous
/// signal `FirstOrderHold` (then within the error of the linear interpolation, `(ω ts)^2 / 12`
/// relative at the frequency `ω`; for the highest derivative `x^(n) = v - Σ a_i x^(n-i)`, a
/// difference of terms of the order of `v`, that error is relative to `v`, not to `x^(n)`). Unlike `s^i / A(s)` discretized as transfer functions (whose
/// coefficients, from `n` poles near `z = 1` when `λ ts` is small, lose the low-frequency response
/// to rounding at high orders), the states are kept in the time domain:
///
/// ```
/// use dsmc::discretize::{InterSample, StateVariableFilter};
///
/// // F(s) = (λ / (s + λ))^5 at 2 Hz, sampled at 10 kHz
/// let (lambda, ts) = (2.0 * std::f64::consts::PI * 2.0, 1e-4);
/// let filter = StateVariableFilter::lag(5, lambda, ts);
/// let u = vec![1.0; 20000]; // unit step, held (2 s)
/// let x = filter.apply_columns(&u, InterSample::ZeroOrderHold);
/// // F u = λ^5 x, s F u = λ^5 x' (pseudo-derivative)
/// let gain = lambda.powi(5);
/// assert!((gain * x[0][19999] - 1.0).abs() < 1e-3);
/// assert!(gain * x[1][19999] < 1e-2);
/// ```
///
/// The coefficients `a_i` grow like `ρ^i` (`ρ`: radius of the roots), so the companion matrix in
/// the states `x^(i)` spans many decades at high orders and its exponential is inaccurate. The
/// states are therefore scaled as in the normalized time `ρ t`: `z_i = x^(i) / ρ^i`, with the
/// input `w = v / ρ^n`, `ż = ρ (A' z + B w)`, `A'` the companion matrix of `a_i / ρ^i`.
#[derive(Clone, Debug)]
pub struct StateVariableFilter<T> {
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
    ///
    /// `ts` is the sampling period \[s\]. `a = []` (`A = 1`) is the identity: no state, `apply`
    /// gives the column `[v]`.
    pub fn new(a: &[T], ts: T) -> Self {
        let n = a.len();
        let rho = root_radius(a);
        if n == 0 {
            return Self { a: Vec::new(), rho, phi: DMatrix::zeros(0, 0), gamma0: DVector::zeros(0), gamma1: DVector::zeros(0) };
        }
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

    /// `1 / (s + λ)^order` (`lambda` = `λ` \[rad/s\]): `a_k = C(order, k) λ^k`. `λ^order` times
    /// the outputs are those of the low-pass `F(s) = (λ / (s + λ))^order` and its
    /// pseudo-derivatives `s^i F(s)`.
    pub fn lag(order: usize, lambda: T, ts: T) -> Self {
        // (s + λ)^order, one factor at a time: c_k += λ c_(k-1)
        let mut c = vec![T::one()];
        for _ in 0..order {
            c.push(T::zero());
            for k in (1..c.len()).rev() {
                c[k] = c[k] + lambda * c[k - 1];
            }
        }
        Self::new(&c[1..], ts)
    }

    /// Order `n` of `A(s)`.
    pub fn order(&self) -> usize {
        self.a.len()
    }

    /// `[x, x', ..., x^(n)]` of `x = v / A(s)` at the sampling instants, from rest, for `v`
    /// sampled with the period given to `new` and of the intersample behaviour `hold`: row `k`,
    /// column `i` is `x^(i)[k]` (`v.len()` rows, `n + 1` columns).
    pub fn apply(&self, v: &[T], hold: InterSample) -> DMatrix<T> {
        let n = self.a.len();
        let scale: Vec<T> = (0..=n).map(|i| Float::powi(self.rho, i as i32)).collect();
        let mut out = DMatrix::zeros(v.len(), n + 1);
        // State and next state, reused (no allocation per sample)
        let (mut z, mut next) = (DVector::zeros(n), DVector::zeros(n));
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
            // z[k+1] = Φ z[k] + Γ0 w[k] + Γ1 (w[k+1] - w[k])
            next.gemv(T::one(), &self.phi, &z, T::zero());
            next.axpy(v[k] / scale[n], &self.gamma0, T::one());
            next.axpy(slope / scale[n], &self.gamma1, T::one());
            std::mem::swap(&mut z, &mut next);
        }
        out
    }

    /// `apply` by columns: element `i` is `x^(i)` over the samples (`n + 1` vectors of
    /// `v.len()`).
    pub fn apply_columns(&self, v: &[T], hold: InterSample) -> Vec<Vec<T>> {
        let out = self.apply(v, hold);
        out.column_iter().map(|c| c.iter().copied().collect()).collect()
    }
}

/// Radius of the roots of the monic `s^n + a_1 s^(n-1) + ... + a_n`, `max_i |a_i|^(1/i)` (1 if all
/// `a_i` are zero).
pub(crate) fn root_radius<T: Float>(a: &[T]) -> T {
    let rho = a.iter().enumerate().fold(T::zero(), |acc, (i, &c)| acc.max(c.abs().powf(T::one() / T::from(i + 1).unwrap())));
    if rho > T::zero() { rho } else { T::one() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_without_poles() {
        let v = [1.0, -2.0, 0.5, 3.0];
        for hold in [InterSample::ZeroOrderHold, InterSample::FirstOrderHold] {
            let out = StateVariableFilter::<f64>::new(&[], 1e-3).apply(&v, hold);
            assert_eq!(out.shape(), (4, 1));
            assert_eq!(out.column(0).as_slice(), &v);
        }
    }
}
