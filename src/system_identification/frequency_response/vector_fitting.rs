//! Vector Fitting (VF) method.

use std::iter::Sum;

use nalgebra::{Complex, DMatrix, DVector, RealField};

use num_traits::Float;
use thiserror::Error;

use crate::{Continuous, FrequencyResponse, Polynomial, TransferFunction};

/// Errors of vector fitting.
#[derive(Error, Debug)]
pub enum VectorFittingError {
    /// No samples.
    #[error("No sampled data provided")]
    EmptyData,
    /// A least-squares problem is singular.
    #[error("Matrix is singular or numerically unstable (check order and data)")]
    SingularMatrix,
    /// No poles to fit with.
    #[error("Poles not set")]
    PolesNotSet,
    /// The zeros of `σ(s)` (new poles) could not be computed.
    #[error("Failed to find zeros")]
    ZerosNotFound,
    /// The iteration failed.
    #[error("Failed to iterate (check data and iterations)")]
    IterationError,
    /// The weights are not one per sample, or not all finite and non-negative.
    #[error("Invalid weights: {len} for {samples} samples, all finite and non-negative needed")]
    InvalidWeights { len: usize, samples: usize },
}

/// Options of vector fitting.
#[derive(Debug, Clone)]
pub struct VectorFittingOptions<T> {
    /// Maximum number of pole relocation iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: T,
    /// Fit the constant term `d`.
    pub fit_d: bool,
    /// Fit the proportional term `e s`.
    pub fit_e: bool,
}

impl<T: Float> Default for VectorFittingOptions<T> {
    fn default() -> Self {
        Self {
            max_iter: 10,
            tol: T::from(1e-8).unwrap(),
            fit_d: false,
            fit_e: false,
        }
    }
}

/// Fitted model `G(s) = Σ r_k / (s - p_k) + d + e s`. Converts into a `TransferFunction` with `into`.
///
/// From `identify`, the model has real coefficients: the poles are real or in conjugate pairs
/// (the one with `Im p > 0` first, then its conjugate), the residues of a real pole are real and
/// those of a pair conjugate, exactly.
#[derive(Debug, Clone)]
pub struct VectorFittingResult<T> {
    /// Poles `p_k`.
    pub poles: Vec<Complex<T>>,
    /// Residues `r_k`, one per pole.
    pub residues: Vec<Complex<T>>,
    /// Constant term.
    pub d: T,
    /// Proportional term (coefficient of `s`).
    pub e: T,
    /// RMS fitting error after each iteration (of the weighted error by `identify_weighted`).
    pub rms_errors: Vec<T>,
}

/// A real pole, or a pair of conjugate poles `p`, `p*` (`Im p > 0`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Block<T> {
    Real(T),
    Pair(Complex<T>),
}

impl<T: Float> Block<T> {
    /// Number of poles (and of real residue parameters): 1 or 2.
    fn size(&self) -> usize {
        match self {
            Block::Real(_) => 1,
            Block::Pair(_) => 2,
        }
    }

    /// The pole with `Im p >= 0`.
    fn pole(&self) -> Complex<T> {
        match *self {
            Block::Real(p) => Complex::new(p, T::zero()),
            Block::Pair(p) => p,
        }
    }

    /// Real basis at `s`: `1 / (s - p)` for a real pole; `1 / (s - p) + 1 / (s - p*)` and
    /// `j / (s - p) - j / (s - p*)` for a pair, so that the real parameters `c'`, `c''` stand for
    /// the conjugate residues `c' ± j c''`.
    fn basis(&self, s: Complex<T>) -> [Complex<T>; 2] {
        match *self {
            Block::Real(p) => [Complex::from(T::one()) / (s - p), Complex::from(T::zero())],
            Block::Pair(p) => {
                let (u, v) = (Complex::from(T::one()) / (s - p), Complex::from(T::one()) / (s - p.conj()));
                let j = Complex::new(T::zero(), T::one());
                [u + v, j * (u - v)]
            }
        }
    }
}

/// Poles of `blocks` (each pair as `p`, `p*`).
fn poles_of<T: Float>(blocks: &[Block<T>]) -> Vec<Complex<T>> {
    blocks
        .iter()
        .flat_map(|b| match *b {
            Block::Real(p) => vec![Complex::new(p, T::zero())],
            Block::Pair(p) => vec![p, p.conj()],
        })
        .collect()
}

/// Complex residues of the real parameters `c` (`c' ± j c''` for a pair), one per pole.
fn residues_of<T: Float>(blocks: &[Block<T>], c: &[T]) -> Vec<Complex<T>> {
    let mut residues = Vec::with_capacity(c.len());
    let mut i = 0;
    for b in blocks {
        match b {
            Block::Real(_) => residues.push(Complex::new(c[i], T::zero())),
            Block::Pair(_) => {
                residues.push(Complex::new(c[i], c[i + 1]));
                residues.push(Complex::new(c[i], -c[i + 1]));
            }
        }
        i += b.size();
    }
    residues
}

/// Real model of arbitrary poles and residues, as `(blocks, c)`: a real pole keeps the real part
/// of its residue; a complex pole is paired with the pole closest to its conjugate (within
/// `1e-6 |p|`), with the mean `(r_p + r_q*) / 2` as its residue; a complex pole without its
/// conjugate is completed by it (a real model has both), with the conjugate residue. For a
/// result of `identify` (real poles and exact conjugate pairs) this is exact.
fn real_model<T: Float>(poles: &[Complex<T>], residues: &[Complex<T>]) -> (Vec<Block<T>>, Vec<T>) {
    let two = T::from(2).unwrap();
    let mut used = vec![false; poles.len()];
    let (mut blocks, mut c) = (Vec::new(), Vec::new());
    for i in 0..poles.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let (p, r) = (poles[i], residues[i]);
        if p.im == T::zero() {
            blocks.push(Block::Real(p.re));
            c.push(r.re);
            continue;
        }
        let partner = (0..poles.len())
            .filter(|&j| !used[j] && poles[j].im != T::zero())
            .min_by(|&a, &b| (poles[a] - p.conj()).norm().partial_cmp(&(poles[b] - p.conj()).norm()).unwrap_or(std::cmp::Ordering::Equal))
            .filter(|&j| (poles[j] - p.conj()).norm() <= T::from(1e-6).unwrap() * p.norm());
        // Residue of the pole with Im > 0
        let (upper, residue) = match partner {
            Some(j) => {
                used[j] = true;
                let mean = (r + residues[j].conj()) / two;
                if p.im > T::zero() { (p, mean) } else { (p.conj(), mean.conj()) }
            }
            None if p.im > T::zero() => (p, r),
            None => (p.conj(), r.conj()),
        };
        blocks.push(Block::Pair(upper));
        c.push(residue.re);
        c.push(residue.im);
    }
    (blocks, c)
}

/// Product of polynomials (descending coefficients).
fn multiply<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    let mut out = vec![T::zero(); a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] = out[i + j] + x * y;
        }
    }
    out
}

/// `b` added to `a`, aligned at the constant term.
fn add_into<T: Float>(a: &mut [T], b: &[T]) {
    let offset = a.len() - b.len();
    for (i, &c) in b.iter().enumerate() {
        a[offset + i] = a[offset + i] + c;
    }
}

impl<T: Float + RealField> From<VectorFittingResult<T>> for TransferFunction<T, Continuous> {
    /// `G(s) = B(s) / A(s)` in real arithmetic, from the real poles and the conjugate pairs: a real
    /// pole gives the factor `s - p` and the term `c`, a pair `s^2 - 2 a' s + |p|^2` and
    /// `2 c' s - 2 (c' a' + c'' a'')` (residues `c' ± j c''`, `p = a' + j a''`). Poles and
    /// residues that are not those of a real model are made so first (see `VectorFittingResult`;
    /// a real pole keeps the real part of its residue, a pair the mean of `r_p` and `r_p*^*`).
    fn from(val: VectorFittingResult<T>) -> Self {
        let n = val.poles.len();
        if n == 0 {
            return TransferFunction::from_polynomials(
                Polynomial(vec![val.e, val.d]),
                Polynomial(vec![T::one()]),
            );
        }

        let two = T::from(2).unwrap();
        let (blocks, c) = real_model(&val.poles, &val.residues);
        let factors: Vec<Vec<T>> = blocks
            .iter()
            .map(|b| match *b {
                Block::Real(p) => vec![T::one(), -p],
                Block::Pair(p) => vec![T::one(), -two * p.re, p.norm_sqr()],
            })
            .collect();
        let denom = factors.iter().fold(vec![T::one()], |acc, f| multiply(&acc, f));

        // B(s) = Σ_b B_b(s) Π_{other} A_b'(s) + A(s) d + s A(s) e
        let degree = denom.len() - 1;
        let mut numer = vec![T::zero(); degree + 2];
        let mut i = 0;
        for (k, b) in blocks.iter().enumerate() {
            let term = match *b {
                Block::Real(_) => vec![c[i]],
                Block::Pair(p) => vec![two * c[i], -two * (c[i] * p.re + c[i + 1] * p.im)],
            };
            let others = factors.iter().enumerate().filter(|&(j, _)| j != k).fold(vec![T::one()], |acc, (_, f)| multiply(&acc, f));
            add_into(&mut numer, &multiply(&term, &others));
            i += b.size();
        }
        add_into(&mut numer, &denom.iter().map(|&a| a * val.d).collect::<Vec<T>>());
        let mut s_denom: Vec<T> = denom.iter().map(|&a| a * val.e).collect();
        s_denom.push(T::zero());
        add_into(&mut numer, &s_denom);

        // Drop the leading coefficients that are negligible (d, e not fitted, or a residue sum
        // that cancels): the term c_i s^m is compared at the largest pole magnitude ω_ref with
        // the largest term, so that the threshold does not depend on the units of G nor on the
        // frequency scale
        let omega_ref = val.poles.iter().map(|p| p.norm()).fold(T::zero(), Float::max);
        let omega_ref = if omega_ref > T::zero() { omega_ref } else { T::one() };
        let top = numer.len() - 1;
        let terms: Vec<T> = numer.iter().enumerate().map(|(i, c)| Float::abs(*c) * Float::powi(omega_ref, (top - i) as i32)).collect();
        let largest = terms.iter().copied().fold(T::zero(), Float::max);
        let first_valid = terms.iter()
            .position(|&t| t > T::from(1e-5).unwrap() * largest)
            .unwrap_or(numer.len());
        numer.drain(..first_valid);

        TransferFunction::from_polynomials(Polynomial(numer), Polynomial(denom))
    }
}

/// Fit `n_poles` poles to the samples (`omega` in rad/s) by vector fitting; robust for high orders
/// and resonant systems.
///
/// Reference: B. Gustavsen and A. Semlyen,
/// "Rational approximation of frequency domain responses by Vector Fitting",
/// IEEE Trans. Power Delivery, vol. 14, no. 3, pp. 1052-1061, July 1999.
///
/// Model:
///
/// ```text
/// H(s) ≈ Σ_k [ c_k / (s - a_k) ] + d + s*e
/// ```
///
/// Algorithm:
///
/// 1. Set initial poles `{a_k}` (complex pairs logarithmically spaced over the band, lightly
///    damped; a real pole for an odd order)
/// 2. Solve the least-squares problem, over the samples `s = jω` (real and imaginary parts)
///    ```text
///    σ(s)・H(s) ≈ Σ c_k/(s-a_k) + d + s e,   σ(s) = 1 + Σ c̃_k/(s-a_k)
///    ```
/// 3. Set the zeros of `σ(s)` as the new poles `{a_k}` (reflected to the left half-plane)
/// 4. Repeat 2-3 until convergence
/// 5. After convergence, solve for the final residues `{c_k}`, `d`, `e` by least squares
///
/// The model is kept real (Appendix of the reference): the unknowns are real, `c` for a real
/// pole and `c'`, `c''` for a pair `a, a*`, with the basis `1 / (s - a) + 1 / (s - a*)` and
/// `j / (s - a) - j / (s - a*)` (residues `c' ± j c''`), so that the residues of a pair are
/// conjugate and those of a real pole real, also on noisy data (independent complex residues fit
/// the noise of the positive frequencies with a model that is not real, and the iterations
/// oscillate). The zeros of `σ` are the eigenvalues of the real matrix `H = A - b c̃ᵀ`, `A` block
/// diagonal (`a` for a real pole, `[[a', a''], [-a'', a']]` for a pair `a' ± j a''`) and `b` of
/// blocks `1` and `[2, 0]`: real, or exact conjugate pairs. The columns of the least-squares
/// problems are scaled to unit norm and solved by SVD.
///
/// The error is absolute: on a response spanning decades, the bins of the largest gain dominate.
/// `identify_weighted` weights it, e.g. by `1 / |G|` for the relative error.
pub fn identify<T: Float + RealField + Sum>(
    samples: &[FrequencyResponse<T>],
    n_poles: usize,
    opts: &VectorFittingOptions<T>,
) -> Result<VectorFittingResult<T>, VectorFittingError> {
    fit(samples, None, n_poles, opts)
}

/// `identify` minimizing the weighted error `Σ w_i^2 |G_fit(jω_i) - G_i|^2` (also in the pole
/// relocation, where the error of each sample is `σ H - (Σ c / (s - a) + d + s e)`), with one
/// finite, non-negative weight per sample: e.g. `w_i = 1 / |G_i|` for the relative error, or the
/// inverse standard deviation of each bin. `rms_errors` are those of the weighted error.
pub fn identify_weighted<T: Float + RealField + Sum>(
    samples: &[FrequencyResponse<T>],
    weights: &[T],
    n_poles: usize,
    opts: &VectorFittingOptions<T>,
) -> Result<VectorFittingResult<T>, VectorFittingError> {
    if weights.len() != samples.len() || weights.iter().any(|w| !(w.is_finite() && *w >= T::zero())) {
        return Err(VectorFittingError::InvalidWeights { len: weights.len(), samples: samples.len() });
    }
    fit(samples, Some(weights), n_poles, opts)
}

fn fit<T: Float + RealField + Sum>(
    samples: &[FrequencyResponse<T>],
    weights: Option<&[T]>,
    n_poles: usize,
    opts: &VectorFittingOptions<T>,
) -> Result<VectorFittingResult<T>, VectorFittingError> {
    if samples.is_empty() {
        return Err(VectorFittingError::EmptyData);
    }
    let weight = |i: usize| weights.map_or(T::one(), |w| w[i]);

    // ---- Step 1: Set initial poles ----
    let mut blocks = initial_blocks(samples, n_poles);
    let mut rms_errors = Vec::new();

    for _iter in 0..opts.max_iter {
        // ---- Step 2: Construct a least-squares problem and solve it ----
        let (c, d, e, c_tilde) = least_squares(samples, &weight, &blocks, opts, true)?;
        rms_errors.push(rms(samples, &weight, &blocks, &c, d, e));

        // ---- Step 3: Zeros of σ(s) as the new poles ----
        let new_blocks = zeros_of_sigma(&blocks, &c_tilde)?;

        // Convergence check (blocks sorted alike; a change of structure is not converged)
        let epsilon = T::from(1e-30).unwrap();
        let max_change = if new_blocks.len() == blocks.len() && new_blocks.iter().zip(&blocks).all(|(a, b)| a.size() == b.size()) {
            blocks
                .iter()
                .zip(new_blocks.iter())
                .map(|(old, new)| (new.pole() - old.pole()).norm() / (old.pole().norm() + epsilon))
                .fold(T::zero(), Float::max)
        } else {
            T::infinity()
        };
        blocks = new_blocks;
        if max_change < opts.tol {
            break;
        }
    }

    // ---- Step 5: Final residues and constant terms ----
    let (c, d, e, _) = least_squares(samples, &weight, &blocks, opts, false)?;
    Ok(VectorFittingResult { poles: poles_of(&blocks), residues: residues_of(&blocks, &c), d, e, rms_errors })
}

/// Complex pairs `-0.01 ω ± j ω` logarithmically spaced over the band, and a real pole `-ω_mid`
/// for an odd order.
fn initial_blocks<T: Float>(samples: &[FrequencyResponse<T>], n_poles: usize) -> Vec<Block<T>> {
    let w_min = samples.iter().fold(T::infinity(), |a, b| T::min(a, b.omega));
    let w_max = samples.iter().fold(T::neg_infinity(), |a, b| T::max(a, b.omega));

    let n_pairs = n_poles / 2;
    let mut blocks = Vec::with_capacity(n_pairs + 1);
    for i in 0..n_pairs {
        let t = if n_pairs > 1 {
            T::from(i as f64 / (n_pairs - 1) as f64).unwrap()
        } else {
            T::from(0.5).unwrap()
        };
        let w = w_min * (w_max / w_min).powf(t);
        blocks.push(Block::Pair(Complex::new(-T::from(0.01).unwrap() * w, w)));
    }
    if n_poles % 2 == 1 {
        blocks.push(Block::Real(-(w_min * w_max).sqrt()));
    }
    blocks
}

/// Real parameters `(c, d, e, c̃)` of the least-squares problem over the samples (rows: the real
/// and the imaginary parts of each weighted sample):
///
/// ```text
/// with_sigma:  Σ c φ(s) + d + s e - G Σ c̃ φ(s) = G
/// otherwise:   Σ c φ(s) + d + s e = G                 (c̃ empty)
/// ```
///
/// `φ` the real basis of the blocks. The columns are scaled to unit norm before the SVD solve.
#[allow(clippy::type_complexity)]
fn least_squares<T: Float + RealField>(
    samples: &[FrequencyResponse<T>],
    weight: &impl Fn(usize) -> T,
    blocks: &[Block<T>],
    opts: &VectorFittingOptions<T>,
    with_sigma: bool,
) -> Result<(Vec<T>, T, T, Vec<T>), VectorFittingError> {
    let n: usize = blocks.iter().map(Block::size).sum();
    let n_extra = opts.fit_d as usize + opts.fit_e as usize;
    let columns = n + n_extra + if with_sigma { n } else { 0 };
    let rows = 2 * samples.len();
    let mut a = DMatrix::<T>::zeros(rows, columns);
    let mut b = DVector::<T>::zeros(rows);

    for (i, sample) in samples.iter().enumerate() {
        let w = weight(i);
        let s = Complex::new(T::zero(), sample.omega);
        let g = sample.value;
        let mut row = Vec::with_capacity(columns);
        for block in blocks {
            row.extend_from_slice(&block.basis(s)[..block.size()]);
        }
        if opts.fit_d {
            row.push(Complex::from(T::one()));
        }
        if opts.fit_e {
            row.push(s);
        }
        if with_sigma {
            for block in blocks {
                row.extend(block.basis(s)[..block.size()].iter().map(|&phi| -g * phi));
            }
        }
        for (j, v) in row.iter().enumerate() {
            a[(2 * i, j)] = w * v.re;
            a[(2 * i + 1, j)] = w * v.im;
        }
        b[2 * i] = w * g.re;
        b[2 * i + 1] = w * g.im;
    }

    let scale: Vec<T> = (0..columns)
        .map(|j| {
            let norm = a.column(j).norm();
            if norm > T::zero() { norm } else { T::one() }
        })
        .collect();
    for (j, &sc) in scale.iter().enumerate() {
        a.column_mut(j).unscale_mut(sc);
    }
    let x = a
        .svd(true, true)
        .solve(&b, T::from(1e-12).unwrap())
        .map_err(|_| VectorFittingError::SingularMatrix)?;
    let x: Vec<T> = x.iter().zip(&scale).map(|(&v, &sc)| v / sc).collect();
    if x.iter().any(|v| !v.is_finite()) {
        return Err(VectorFittingError::SingularMatrix);
    }

    let c = x[..n].to_vec();
    let mut k = n;
    let d = if opts.fit_d { k += 1; x[k - 1] } else { T::zero() };
    let e = if opts.fit_e { k += 1; x[k - 1] } else { T::zero() };
    let c_tilde = x[k..].to_vec();
    Ok((c, d, e, c_tilde))
}

/// Zeros of `σ(s) = 1 + Σ c̃ φ(s)` (real parameters `c̃` of the blocks) as the eigenvalues of
/// the real matrix `H = A - b c̃ᵀ` (Gustavsen and Semlyen, Appendix): `A` block diagonal with `a`
/// for a real pole and `[[a', a''], [-a'', a']]` for a pair `a' ± j a''`, `b` of blocks `1` and
/// `[2, 0]`. The eigenvalues of a real matrix are real or exact conjugate pairs; those with
/// `|Im| <= 1e-9 |λ|` are taken as real. Poles in the right half-plane are reflected to the left
/// one; the blocks are sorted by the magnitude of the pole.
fn zeros_of_sigma<T: Float + RealField>(blocks: &[Block<T>], c_tilde: &[T]) -> Result<Vec<Block<T>>, VectorFittingError> {
    let n = c_tilde.len();
    if n == 0 {
        return Err(VectorFittingError::PolesNotSet);
    }
    let mut h = DMatrix::<T>::zeros(n, n);
    let mut b = DVector::<T>::zeros(n);
    let mut i = 0;
    for block in blocks {
        match *block {
            Block::Real(p) => {
                h[(i, i)] = p;
                b[i] = T::one();
            }
            Block::Pair(p) => {
                h[(i, i)] = p.re;
                h[(i, i + 1)] = p.im;
                h[(i + 1, i)] = -p.im;
                h[(i + 1, i + 1)] = p.re;
                b[i] = T::from(2).unwrap();
            }
        }
        i += block.size();
    }
    for r in 0..n {
        for col in 0..n {
            h[(r, col)] -= b[r] * c_tilde[col];
        }
    }

    let eigenvalues = nalgebra::linalg::Schur::try_new(h, T::default_epsilon(), 0)
        .ok_or(VectorFittingError::ZerosNotFound)?
        .complex_eigenvalues();
    let mut new_blocks = Vec::with_capacity(n);
    for z in eigenvalues.iter() {
        let z = Complex::new(z.re, z.im);
        if !(z.re.is_finite() && z.im.is_finite()) {
            return Err(VectorFittingError::ZerosNotFound);
        }
        let re = -Float::abs(z.re);
        if Float::abs(z.im) <= T::from(1e-9).unwrap() * z.norm() {
            new_blocks.push(Block::Real(re));
        } else if z.im > T::zero() {
            new_blocks.push(Block::Pair(Complex::new(re, z.im)));
        }
    }
    if new_blocks.iter().map(Block::size).sum::<usize>() != n {
        return Err(VectorFittingError::ZerosNotFound);
    }
    new_blocks.sort_by(|a, b| a.pole().norm().partial_cmp(&b.pole().norm()).unwrap_or(std::cmp::Ordering::Equal));
    Ok(new_blocks)
}

/// RMS of the (weighted) error of `Σ c φ + d + s e` over the samples.
fn rms<T: Float + RealField + Sum>(
    samples: &[FrequencyResponse<T>],
    weight: &impl Fn(usize) -> T,
    blocks: &[Block<T>],
    c: &[T],
    d: T,
    e: T,
) -> T {
    let n = T::from(samples.len()).unwrap();
    let sum_sq: T = samples
        .iter()
        .enumerate()
        .map(|(i, sample)| {
            let s = Complex::new(T::zero(), sample.omega);
            let mut fit = Complex::new(d, T::zero()) + s * e;
            let mut k = 0;
            for block in blocks {
                for phi in &block.basis(s)[..block.size()] {
                    fit += *phi * c[k];
                    k += 1;
                }
            }
            (fit - sample.value).norm_sqr() * weight(i) * weight(i)
        })
        .sum();
    Float::sqrt(sum_sq / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    /// The eigenvalues of `H` are the zeros of `σ(s) = 1 + Σ r_k / (s - a_k)` with the complex
    /// residues of the real parameters.
    #[test]
    fn test_zeros_of_sigma() {
        let blocks = vec![Block::Real(-3.0), Block::Pair(Complex64::new(-1.0, 2.0)), Block::Pair(Complex64::new(-0.5, 10.0))];
        let c_tilde = vec![0.7, 1.0, -0.4, 2.0, 0.3];
        let poles = poles_of(&blocks);
        let residues = residues_of(&blocks, &c_tilde);
        assert_eq!(residues[1], residues[2].conj());

        let zeros = zeros_of_sigma(&blocks, &c_tilde).unwrap();
        assert_eq!(zeros.iter().map(Block::size).sum::<usize>(), 5);
        for z in poles_of(&zeros) {
            // Reflected zeros: σ vanishes at the zero or at its mirror image
            let sigma = |z: Complex64| residues.iter().zip(&poles).fold(Complex64::new(1.0, 0.0), |acc, (r, p)| acc + r / (z - p));
            let value = sigma(z).norm().min(sigma(Complex64::new(-z.re, z.im)).norm());
            assert!(value < 1e-8, "σ({z}) = {value:.2e}");
        }
    }

    /// The real basis of a pair is `(c' + j c'') / (s - p) + (c' - j c'') / (s - p*)`.
    #[test]
    fn test_pair_basis() {
        let p = Complex64::new(-1.0, 3.0);
        let s = Complex64::new(0.0, 2.0);
        let (c1, c2) = (0.4, -1.5);
        let [phi1, phi2] = Block::Pair(p).basis(s);
        let r = Complex64::new(c1, c2);
        let expected = r / (s - p) + r.conj() / (s - p.conj());
        assert!((phi1 * c1 + phi2 * c2 - expected).norm() < 1e-14);
    }
}
