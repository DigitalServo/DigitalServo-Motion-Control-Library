// Vector Fitting (VF) method
//
// Reference: B. Gustavsen and A. Semlyen,
//   "Rational approximation of frequency domain responses by Vector Fitting"
//   IEEE Trans. Power Delivery, vol. 14, no. 3, pp. 1052-1061, July 1999.
//
// Model:
//   H(s) ≈ Σ_k [ c_k / (s - a_k) ] + d + s*e
//
// Algorithm:
//   1. Set initial poles {a_k} (logarithmic spacing on the imaginary axis)
//   2. Solve weighted least-squares problem
//      σ(s)・H(s) ≈ Σ c̃_k/(s-a_k) + d̃  (lhs)
//      σ(s)       ≈ Σ ĉ_k/(s-a_k) + 1    (rhs constraint)
//   3. Find zeros of σ(s) to find new poles {a_k}
//      → set it to {a_k}
//   4. Repeat 2-3 until convergence
//   5. After convergence, solve for the final residues {c_k}, d using least squares

use std::{iter::Sum, ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign}};

use nalgebra::{Complex, DMatrix, DVector, RealField};

use num_traits::Float;
use thiserror::Error;

use crate::{FrequencyResponse, Polynomial, TransferFunction};

#[derive(Error, Debug)]
pub enum VectorFittingError {
    #[error("No sampled data provided")]
    EmptyData,
    #[error("Matrix is singular or numerically unstable (check order and data)")]
    SingularMatrix,
    #[error("Poles not set")]
    PolesNotSet,
    #[error("Failed to find zeros")]
    ZerosNotFound,
    #[error("Failed to iterate (check data and iterations)")]
    IterationError,
}

/// [Helper function] poly_from_roots: roots → monic polynomial (descending order)
/// e.g. roots=[-1,-2] → [1, 3, 2] (= (s+1)(s+2))
fn poly_from_roots<T: Float + RealField>(roots: &[Complex<T>]) -> Vec<Complex<T>> {
    let mut p = vec![Complex::new(T::one(), T::zero())];
    for &r in roots {
        let mut q = vec![Complex::from(T::zero()); p.len() + 1];
        for (i, &c) in p.iter().enumerate() {
            q[i] += c;          // c * s
            q[i + 1] -= c * r;  // -c * r
        }
        p = q;
    }
    p
}

#[derive(Debug, Clone)]
pub struct VectorFittingOptions<T> {
    pub max_iter: usize,
    pub tol: T,
    pub fit_d: bool,
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

#[derive(Debug, Clone)]
pub struct VectorFittingResult<T> {
    pub poles: Vec<Complex<T>>,
    pub residues: Vec<Complex<T>>,
    pub d: T,
    pub e: T,
    pub rms_errors: Vec<T>,
}

impl<T: Float + RealField> Into<TransferFunction<T>> for VectorFittingResult<T> {
    fn into(self) -> TransferFunction<T> {
        let n = self.poles.len();
        if n == 0 {
            return TransferFunction {
                numerator: Polynomial(vec![self.e, self.d]),
                denominator: Polynomial(vec![T::one()]),
            };
        }

        // G(s) = B(s) / A(s)
        // A(s) = ∏_{k=1 to N}(s - p_k)
        // B(s) = Σ_{k=1 to N} [r_k* (∏_{j!=k}(s - p_j))] + A(s)d + sA(s)e = numer1 + numer2 + numer3

        // Compute A(s)
        let denom_complex = poly_from_roots(&self.poles);
        let denom: Vec<T> = denom_complex.iter().map(|c| c.re).collect();

        // Compute B(s)
        let numer: Vec<T> = {
            // The maximum length of the numerator polynomial is n + 2;
            // The highest order tem depends on s*A(s)*e.
            let numer_len = self.poles.len() + 2;
            let mut numer_complex: Vec<Complex<T>> = vec![Complex::from(T::zero()); numer_len];

            // numer1
            for k in 0..n {
                let other_poles: Vec<_> = (0..n)
                    .filter(|&j| j != k)
                    .map(|j| self.poles[j])
                    .collect();

                let r_k = self.residues[k];
                let sub_poly: Vec<Complex<T>> = poly_from_roots(&other_poles)
                    .iter()
                    .map(|c| r_k * c)
                    .collect();

                // Right-shift
                let offset = numer_complex.len() - sub_poly.len();
                for (i, &c) in sub_poly.iter().enumerate() {
                    numer_complex[offset + i] += c;
                }
            }

            // numer2
            {
                let sub_poly: Vec<Complex<T>> = denom_complex
                    .iter()
                    .map(|c| c * Complex::from(self.d))
                    .collect();

                // Right-shift
                let offset = numer_complex.len() - sub_poly.len();
                for (i, &c) in sub_poly.iter().enumerate() {
                    numer_complex[offset + i] += c;
                }
            }

            // numer3
            {
                let mut sub_poly: Vec<Complex<T>> = denom_complex
                    .iter()
                    .map(|c| c * Complex::from(self.e))
                    .collect();
                sub_poly.push(Complex::from(T::zero()));

                // Right-shift
                let offset = numer_complex.len() - sub_poly.len();
                for (i, &c) in sub_poly.iter().enumerate() {
                    numer_complex[offset + i] += c;
                }
            }

            let mut numer: Vec<T> = numer_complex.iter().map(|c| c.re).collect();

            let first_valid = numer.iter()
                .position(|x| x.abs() >= T::from(1e-5).unwrap())
                .unwrap_or(numer.len());

            numer.drain(..first_valid);

            numer
        };

        TransferFunction {
            numerator: Polynomial(numer),
            denominator: Polynomial(denom),
        }
    }
}

pub fn identify<T: Float + RealField + Sum>(
    samples: &[FrequencyResponse<T>],
    n_poles: usize,
    opts: &VectorFittingOptions<T>,
) -> Result<VectorFittingResult<T>, VectorFittingError> {
    let n_samples = samples.len();
    if n_samples == 0 {
        return Err(VectorFittingError::EmptyData);
    }

    // ---- Step 1: Set initial poles (logarithmic spacing on the imaginary axis) ----
    let mut poles = generate_initial_poles(&samples, n_poles);

    let mut rms_errors = Vec::new();

    for _iter in 0..opts.max_iter {
        // ---- Step 2: Construct a least-squares problem and solve it ----
        // let (sigma_residues, res, d, e) = solve_normal_equation(&samples, &poles, opts)?;
        let (sigma_residues, res, d, e) = solve_accumulated_normal_equation(&samples, &poles, opts)?;

        let rms = compute_rms(&samples, &poles, &res, d, e);
        rms_errors.push(rms);

        // ---- Step 3: Find zeros of σ(s) to find new poles ----
        let new_poles = zeros_of_sigma(&poles, &sigma_residues)?;


        // Stabilize poles (invert sign if real part is positive)
        let new_poles: Vec<Complex<T>> = new_poles
            .into_iter()
            .map(|p| {
                if p.re > T::zero() {
                    Complex::new(-p.re, p.im)
                } else {
                    p
                }
            })
            .collect();

        // Convergence check
        let epsilon = T::from(1e-30).unwrap();
        let max_change = poles
            .iter()
            .zip(new_poles.iter())
            .map(|(old, new)| (new - old).norm() / (old.norm() + epsilon))
            .fold(T::zero(), Float::max);

        poles = new_poles;

        if max_change < opts.tol {
            break;
        }
    }

    // ---- Step 5: Find final residues and constant terms using least-squares ----
    // let (_, residues, d, e) = solve_normal_equation(&samples, &poles, opts)?;
    let (_, residues, d, e) = solve_accumulated_normal_equation(&samples, &poles, opts)?;

    Ok(VectorFittingResult { poles, residues, d, e, rms_errors })
}


fn generate_initial_poles<T: Float>(samples: &[FrequencyResponse<T>], n_poles: usize) -> Vec<Complex<T>> {
    let w_min = samples.iter().cloned().fold(T::infinity(), |a, b| T::min(a, b.omega));
    let w_max = samples.iter().cloned().fold(T::neg_infinity(), |a, b| T::max(a, b.omega));

    let n_pairs = n_poles / 2;
    let mut poles = Vec::with_capacity(n_poles);

    for i in 0..n_pairs {
        let t = if n_pairs > 1 {
            T::from(i as f64 / (n_pairs - 1) as f64).unwrap()
        } else {
            T::from(0.5).unwrap()
        };
        let w = w_min * (w_max / w_min).powf(t);
        let zeta = T::from(0.01).unwrap();
        let re = -zeta * w;
        poles.push(Complex::new(re,  w));
        poles.push(Complex::new(re, -w));
    }

    if n_poles % 2 == 1 {
        let w_mid = (w_min * w_max).sqrt();
        poles.push(Complex::new(-w_mid, T::zero()));
    }

    poles
}

/// Returns: `(sigma_residues, fitted_residues, d, e)`
#[allow(unused)]
fn solve_normal_equation<T: Float + RealField>(
    samples: &[FrequencyResponse<T>],
    poles: &[Complex<T>],
    opts: &VectorFittingOptions<T>,
) -> Result<(Vec<Complex<T>>, Vec<Complex<T>>, T, T), VectorFittingError> {
    let n = poles.len();
    let n_s = samples.len();
    let n_extra = opts.fit_d as usize + opts.fit_e as usize;
    let parameter_size = 4 * n + n_extra;
    let rows = 2 * n_s;

    let mut a_mat = DMatrix::<T>::zeros(rows, parameter_size);
    let mut b_vec = DVector::<T>::zeros(rows);

    for (i, sample) in samples.iter().enumerate() {
        let si = Complex::new(T::zero(), sample.omega);
        let gi = sample.value;

        let row_re = 2 * i;
        let row_im = 2 * i + 1;

        for k in 0..n {
            let term = gi / (si - poles[k]);
            a_mat[(row_re, 2 * k)]     =  term.re;
            a_mat[(row_re, 2 * k + 1)] = -term.im;
            a_mat[(row_im, 2 * k)]     =  term.im;
            a_mat[(row_im, 2 * k + 1)] =  term.re;
        }

        for k in 0..n {
            let term = -Complex::from(T::one()) / (si - poles[k]);
            a_mat[(row_re, 2 * n + 2 * k)]     =  term.re;
            a_mat[(row_re, 2 * n + 2 * k + 1)] = -term.im;
            a_mat[(row_im, 2 * n + 2 * k)]     =  term.im;
            a_mat[(row_im, 2 * n + 2 * k + 1)] =  term.re;
        }

        let mut col = 4 * n;
        if opts.fit_d {
            a_mat[(row_re, col)] = -T::one();
            a_mat[(row_im, col)] = T::zero();
            col += 1;
        }
        if opts.fit_e {
            a_mat[(row_re, col)] = -si.re;
            a_mat[(row_im, col)] = -si.im;
        }

        b_vec[row_re] = gi.re;
        b_vec[row_im] = gi.im;
    }

    // Solve the problem min‖Aθ + b‖ → Aθ = −b using SVD
    let neg_b = -b_vec;
    let svd = a_mat.svd(true, true);
    let theta = svd
        .solve(&neg_b, T::from(1e-10).unwrap())
        .map_err(|_| VectorFittingError::SingularMatrix)?;

    // c̃_k  (0..2n)
    let tilde_c: Vec<Complex<T>> = (0..n)
        .map(|k| Complex::new(theta[2 * k], theta[2 * k + 1]))
        .collect();

    // c_k  (2n..4n)
    let c: Vec<Complex<T>> = (0..n)
        .map(|k| Complex::new(theta[2 * n + 2 * k], theta[2 * n + 2 * k + 1]))
        .collect();

    // d, h  (4n..)
    let d = if opts.fit_d { theta[4 * n] } else { T::zero() };
    let e = if opts.fit_e { theta[4 * n + 1] } else { T::zero() };

    Ok((tilde_c, c, d, e))
}

/// Returns: `(sigma_residues, fitted_residues, d, e)`
#[allow(unused)]
fn solve_accumulated_normal_equation<T: Float + RealField + AddAssign + MulAssign>(
    samples: &[FrequencyResponse<T>],
    poles: &[Complex<T>],
    opts: &VectorFittingOptions<T>,
) -> Result<(Vec<Complex<T>>, Vec<Complex<T>>, T, T), VectorFittingError> {
    let n = poles.len();
    let num_data = samples.len();
    if num_data == 0 {
        return Err(VectorFittingError::EmptyData);
    }

    let n_extra = opts.fit_d as usize + opts.fit_e as usize;
    let parameter_size = 4 * n + n_extra;    // ˜c (2N), c (2N), d, e
    let mut r = DMatrix::<T>::zeros(parameter_size, parameter_size);
    let mut rhs = DVector::<T>::zeros(parameter_size);

    for sample in samples {
        let s = Complex::new(T::zero(), sample.omega);
        let g = sample.value;

        let mut phi = DMatrix::<T>::zeros(2, parameter_size);

        let mut y = DVector::<T>::zeros(2);
        y[0] = g.re;
        y[1] = g.im;

        for j in 0..n {
            let denom = s - poles[j];

            let term_tilde = g / denom;
            let col_t_re = 2 * j;
            let col_t_im = 2 * j + 1;

            phi[(0, col_t_re)] = term_tilde.re;
            phi[(0, col_t_im)] = -term_tilde.im;
            phi[(1, col_t_re)] = term_tilde.im;
            phi[(1, col_t_im)] = term_tilde.re;

            let term_c = -Complex::from(T::one()) / denom;
            let col_c_re = 2 * n + 2 * j;
            let col_c_im = 2 * n + 2 * j + 1;

            phi[(0, col_c_re)] = term_c.re;
            phi[(0, col_c_im)] = -term_c.im;
            phi[(1, col_c_re)] = term_c.im;
            phi[(1, col_c_im)] = term_c.re;
        }

        if opts.fit_d {
            let col_d = 4 * n;
            phi[(0, col_d)] = -T::one();
            phi[(1, col_d)] = T::zero();
        }

        if opts.fit_e {
            let col_e = 4 * n + 1;
            phi[(0, col_e)] = -s.re;
            phi[(1, col_e)] = -s.im;
        }

        r += &phi.transpose() * &phi;
        rhs += &phi.transpose() * &y;
    }

    let theta = match r.qr().solve(&(-rhs)) {
        Some(x) => x,
        None => return Err(VectorFittingError::SingularMatrix),
    };

    let mut tilde_c = vec![Complex::from(T::zero()); n];
    let mut c = vec![Complex::from(T::zero()); n];

    for j in 0..n {
        let idx_t_re = 2 * j;
        let idx_t_im = 2 * j + 1;
        tilde_c[j] = Complex::new(theta[idx_t_re], theta[idx_t_im]);

        let idx_c_re = 2 * n + 2 * j;
        let idx_c_im = 2 * n + 2 * j + 1;
        c[j] = Complex::new(theta[idx_c_re], theta[idx_c_im]);
    }

    let d = if opts.fit_d { theta[4 * n] } else { T::zero() };
    let e = if opts.fit_e { theta[4 * n + 1] } else { T::zero() };

    Ok((tilde_c, c, d, e))
}


/// Find zeros of σ(s) = 1 + Σ_k c̃_k/(s−a_k)
///
/// ## Algorithm：Real companion matrix
///
/// At first, compute the coefficients of irreducible polynomial N(s),
/// and then solve the eigenvalues of the real companion matrix using nalgebra's Schur decomposition.
///
/// ### Compute coefficients of polynomial
///
/// ```text
/// N(s) = Π_k(s−a_k) + Σ_k c̃_k · Π_{j≠k}(s−a_j)
/// ```
///
/// The coefficients of N(s) are real because because the poles should be real or conjugate pairs.
///
/// ### companion matrix（Frobenius form）
///
/// if N(s) = s^n + p_{n-1}·s^{n-1} + … + p_0：
///
/// ```text
/// C = [0   0  … 0  -p_0  ]
///     [1   0  … 0  -p_1  ]
///     [0   1  … 0  -p_2  ]
///     [⋮       ⋱  ⋮      ]
///     [0   0  … 1  -p_{n-1}]
/// ```
///
/// Eigenvalues of C are the roots of N(s), i.e. the solutions of σ(s)=0.
pub(crate) fn zeros_of_sigma<T: Float + RealField>(
    poles: &[Complex<T>],
    sigma_residues: &[Complex<T>],
) -> Result<Vec<Complex<T>>, VectorFittingError> {
    let n = poles.len();
    if n == 0 {
        return Err(VectorFittingError::PolesNotSet);
    }

    // ---- Step1: N(s) = Π(s−a_k) + Σ_k c̃_k·Π_{j≠k}(s−a_j) ----
    // Coefficients of Π(s−a_k)
    let denom_poly = poly_from_roots(poles);

    // Compute Π_{j≠k}(s−a_j) for each k and multiply by c̃_k, then add to num_poly
    let mut num_poly = denom_poly.clone();

    for k in 0..n {
        let other_poles: Vec<Complex<T>> = (0..n).filter(|&j| j != k).map(|j| poles[j]).collect();
        let sub_poly = poly_from_roots(&other_poles); // 次数 n-1 の多項式

        // Add sub_poly (length: n) to num_poly (length: n+1) and multiply by sigma_residues[k]
        let offset = num_poly.len() - sub_poly.len(); // = 1
        for (i, &c) in sub_poly.iter().enumerate() {
            num_poly[offset + i] += sigma_residues[k] * c;
        }
    }

    // ---- Step2: Abstract the real parts of the coefficients ----
    let real_coeffs: Vec<T> = num_poly.iter().map(|c| c.re).collect();

    // Monic normalize the coefficients (make the leading coefficient 1)
    let lead = real_coeffs[0];
    let monic: Vec<T> = real_coeffs.iter().map(|&c| c / lead).collect();
    // monic = [1, p_{n-1}, …, p_0]（降べきの順）

    // ---- Step3: Construct the companion matrix（Frobenius 形式） ----
    // C[i, n-1] = -monic[n-i]  (i=0..n-1)
    // C[i+1, i] = 1            (i=0..n-2)
    let mut companion = DMatrix::<T>::zeros(n, n);
    for i in 0..n - 1 {
        companion[(i + 1, i)] = T::one();
    }
    for i in 0..n {
        companion[(i, n - 1)] = -monic[n - i];
    }

    // ---- Step4: Get eigen values using Schur decomposition ----
    let schur = nalgebra::linalg::Schur::new(companion);
    let eigs = schur.complex_eigenvalues();
    Ok(eigs.iter().map(|c| Complex::new(c.re, c.im)).collect())
}



fn compute_rms<T: Float + AddAssign + DivAssign + SubAssign + MulAssign + RemAssign + Sum>(
    samples: &[FrequencyResponse<T>],
    poles: &[Complex<T>],
    residues: &[Complex<T>],
    d: T,
    e: T,
) -> T {
    let n = T::from(samples.len()).unwrap();
    let sum_sq: T = samples
        .iter()
        .map(|sample| {
            let sk = Complex::new(T::zero(), sample.omega);
            let mut h_fit = Complex::<T>::new(d, T::zero()) + sk * e;
            for (&a, &c) in poles.iter().zip(residues.iter()) {
                h_fit += c / (sk - a);
            }
            (h_fit - sample.value).norm_sqr()
        })
        .sum();
    (sum_sq / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_zeros_of_sigma() {
        let poles = vec![
            Complex64::new(-1.0,  2.0),
            Complex64::new(-1.0, -2.0),
        ];
        let residues = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let zeros = zeros_of_sigma(&poles, &residues).unwrap();
        println!("Complex case zeros: {:?}", zeros);

        for z in &zeros {
            let sigma_z = residues[0] / (z - poles[0])
                        + residues[1] / (z - poles[1])
                        + Complex64::new(1.0, 0.0);
            assert!(
                sigma_z.norm() < 1e-8,
                "σ({:.4}+{:.4}i) = {:.2e} (should be 0)",
                z.re, z.im, sigma_z.norm()
            );
        }
    }
}
