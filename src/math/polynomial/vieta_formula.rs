use num_complex::Complex;
use num_traits::{Float, One, Zero};

use crate::Polynomial;

/// if roots = [r0, r1, r2,..., rn] is given,
/// that is, p(x) = (x - r0)(x - r1)(x - r2)...(x-rn) = x^(n+1) + an * x^n + an-1 * x^(n-1) + ... + a0,
/// this function returns the coefficient array of p(x) (highest degree first)
/// [1, an, an-1, ..., a0]
pub fn vieta_formula<T: Float>(roots: &[Complex<T>]) -> Polynomial<Complex<T>> {
    let n = roots.len();
    if n == 0 {
        return Polynomial(vec![Complex::one()]);
    }

    // dp[k] = products of sum of k-roots（elementary symmetric sum）
    let mut dp: Vec<Complex<T>> = vec![Complex::zero(); n + 1];
    dp[0] = Complex::one();
    for r in roots.iter() {
        for k in (1..=n).rev() {
            dp[k] = dp[k] + dp[k - 1] * r;
        }
    }

    let mut coeffs: Vec<Complex<T>> = Vec::with_capacity(n + 1);
    coeffs.push(Complex::one());

    for k in 1..=n {
        coeffs.push(if k % 2 == 0 { dp[k] } else { -dp[k] });
    }

    Polynomial(coeffs)
}
