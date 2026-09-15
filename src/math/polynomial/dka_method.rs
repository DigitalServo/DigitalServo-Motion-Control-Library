use num_complex::Complex;
use num_traits::{Float, One, Zero};

use std::f64::consts::PI;

use crate::Polynomial;

pub fn dka_method<T: Float + Zero + One>(coefficients: &Polynomial<Complex<T>>) -> Option<Vec<Complex<T>>> {

    let mut coeffs = coefficients.0.clone();

    while let Some(&c) = coeffs.first() {
        if c != Complex::zero() {
            break;
        }
        coeffs.remove(0);
    }

    let n = coeffs.len();
    if n <= 1 {
        return None;
    }

    /* Normalize the coefficients */
    let leading = coeffs[0];
    let monic_coeffs: Vec<Complex<T>> = coeffs.into_iter().map(|x| x / leading).collect();

    /* Initialize z (Bini / Newton polygon) */
    let z = adaptive_initial_values(&monic_coeffs);
    let z = aberth(&monic_coeffs, z);

    Some(z)
}




/// Upper convex hull of (k, ln|a_k|) over nonzero coeffs, ascending in k.
/// `monic_coeffs` is DESCENDING & monic: [0]=z^degree(=1) ... [degree]=const.
fn newton_polygon_hull<T: Float>(monic_coeffs: &[Complex<T>]) -> Vec<(usize, f64)> {
    let n_len = monic_coeffs.len();
    let degree = n_len - 1;
    let norm_at = |k: usize| monic_coeffs[degree - k].norm(); // ascending a_k

    let mut pts: Vec<(usize, f64)> = Vec::with_capacity(n_len);
    for k in 0..n_len {
        let c = norm_at(k).to_f64().unwrap();
        if c > 0.0 {
            pts.push((k, c.ln()));
        }
    }

    let mut hull: Vec<(usize, f64)> = Vec::new();
    for &p in &pts {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let cross = (b.0 as f64 - a.0 as f64) * (p.1 - a.1)
                      - (b.1 - a.1) * (p.0 as f64 - a.0 as f64);
            if cross >= 0.0 { hull.pop(); } else { break; }
        }
        hull.push(p);
    }
    hull
}


/// Fujiwara-bound initial values. Centroid-centered circle of radius `fujiwara_bound`.
/// `monic_coeffs` is DESCENDING & monic: [0]=z^degree(=1) ... [degree]=const.
fn fujiwara_initial_values<T: Float>(monic_coeffs: &[Complex<T>]) -> Vec<Complex<T>> {

    fn fujiwara_bound<T: Float>(monic_coeffs: &[Complex<T>]) -> T {
        // descending: monic_coeffs[0] = z^degree (=1, monic), monic_coeffs[degree] = const
        let degree = monic_coeffs.len() - 1;
        if degree == 0 {
            return T::one();
        }
        let two = T::from(2.0).unwrap();
        let mut max_val = T::zero();

        // |c_{n-k}|^{1/k} for k = 1..degree-1  → descending index k
        for k in 1..degree {
            let abs_coeff = monic_coeffs[k].norm();
            if abs_coeff > T::zero() {
                let cand = abs_coeff.powf(T::from(1.0 / k as f64).unwrap());
                if cand > max_val { max_val = cand; }
            }
        }
        // constant term: |c_0 / 2|^{1/degree}
        let last = (monic_coeffs[degree].norm() / two)
            .powf(T::from(1.0 / degree as f64).unwrap());
        if last > max_val { max_val = last; }

        two * max_val
    }

    let degree = monic_coeffs.len() - 1;

    let r = fujiwara_bound(monic_coeffs);

    let mut z: Vec<Complex<T>> = Vec::with_capacity(degree);
    for i in 0..degree {
        let angle = T::from(2.0 * PI * (i as f64 + 0.25) / degree as f64).unwrap();
        z.push(Complex::from_polar(r, angle)); // origin-centered
    }

    z
}

/// Bini (Newton polygon) initial values. Origin-centered by construction.
/// `monic_coeffs` is DESCENDING & monic: [0]=z^degree(=1) ... [degree]=const.
fn bini_initial_values<T: Float>(monic_coeffs: &[Complex<T>]) -> Vec<Complex<T>> {
    let n_len = monic_coeffs.len();
    let degree = n_len - 1;
    let two_pi = 2.0 * PI;
    let sigma = 0.7_f64; // fixed offset: keep starts off the real axis

    // ascending coefficient of z^k is monic_coeffs[degree - k]
    let norm_at = |k: usize| monic_coeffs[degree - k].norm();
    let hull = newton_polygon_hull(monic_coeffs);

    let mut z: Vec<Complex<T>> = Vec::with_capacity(degree);

    // constant term == 0 => k_min roots sit exactly at the origin
    let k_min = hull.first().map(|&(k, _)| k).unwrap_or(0);
    if k_min > 0 {
        let tiny = T::from(1e-8).unwrap();
        for l in 0..k_min {
            let angle = sigma + two_pi * (l as f64) / (k_min as f64);
            z.push(Complex::from_polar(tiny, T::from(angle).unwrap()));
        }
    }

    // each hull edge (i -> j): (j-i) roots at radius (|a_i|/|a_j|)^{1/(j-i)}
    for w in hull.windows(2) {
        let (i, _) = w[0];
        let (j, _) = w[1];
        let m = j - i;
        let r = (norm_at(i) / norm_at(j)).powf(T::from(1.0 / m as f64).unwrap());
        let group_rot = two_pi * (i as f64) / (degree as f64);
        for l in 0..m {
            let angle = sigma + group_rot + two_pi * (l as f64) / (m as f64);
            z.push(Complex::from_polar(r, T::from(angle).unwrap()));
        }
    }

    z
}


/// Practical hybrid: choose strategy by modulus spread.
fn adaptive_initial_values<T: Float>(monic_coeffs: &[Complex<T>]) -> Vec<Complex<T>> {
    let hull = newton_polygon_hull(monic_coeffs);
    let edges = hull.len().saturating_sub(1);

    // hull's minimum k > 0 means the constant term vanishes => roots at the origin
    let has_origin_roots = hull.first().map(|&(k, _)| k > 0).unwrap_or(true);

    if edges <= 1 && !has_origin_roots {
        // single modulus band, no origin roots -> Fujiwara (origin-centered, guarantees inclusion, simple) suffices
        fujiwara_initial_values(monic_coeffs)
    } else {
        // multiple bands or origin roots present -> Bini (multi-ring placement)
        bini_initial_values(monic_coeffs)
    }
}


fn aberth<T: Float>(monic_coeffs: &[Complex<T>], mut z: Vec<Complex<T>>) -> Vec<Complex<T>> {

    /// p(x) and p'(x) via Horner on DESCENDING coeffs.
    fn eval_p_dp<T: Float>(desc: &[Complex<T>], x: Complex<T>) -> (Complex<T>, Complex<T>) {
        let mut p  = Complex::zero();
        let mut dp = Complex::zero();
        for &c in desc.iter() {         // highest degree first
            dp = dp * x + p;                        // update derivative with OLD p first
            p  = p * x + c;                         // then update value
        }
        (p, dp)
    }

    let degree = z.len();
    const MAX_ITER: usize = 10000;
    let tol = T::from(1e-12).unwrap();

    for _ in 0..MAX_ITER {
        let mut converged = true;
        let z_old = z.clone();               // Jacobi style
        for i in 0..degree {
            let (p, dp) = eval_p_dp(monic_coeffs, z_old[i]);
            if dp == Complex::zero() { continue; }
            let newton = p / dp;

            let mut s = Complex::zero();
            for j in 0..degree {
                if j != i {
                    let d = z_old[i] - z_old[j];
                    if d != Complex::zero() { s = s + Complex::<T>::one() / d; }
                }
            }
            let denom = Complex::<T>::one() - newton * s;
            if denom == Complex::zero() { continue; }
            let w = newton / denom;

            z[i] = z_old[i] - w;
            let scale = z_old[i].norm().max(T::one());
            if w.norm() > tol * scale { converged = false; }
        }
        if converged { return z; }
    }
    z
}



#[allow(dead_code)]
fn durand_kerner<T: Float>(monic_coeffs: &[Complex<T>], mut z: Vec<Complex<T>>) -> Vec<Complex<T>> {

    fn mapping_h<T: Float + One>(coeffs: &[Complex<T>], x: Complex<T>) -> Complex<T> {
        let mut ret = Complex::zero();
        let mut pow = Complex::one();

        for &c in coeffs.iter().rev() {
            ret = ret + c * pow;
            pow = pow * x;
        }
        ret
    }

    // dh = ∏_{j ≠ i} (z_i - z_j)
    fn mapping_dh<T: Float>(z: &[Complex<T>], i: usize) -> Complex<T> {
        let zi = z[i];
        let mut ret = Complex::one();
        for (j, &zj) in z.iter().enumerate() {
            if i != j {
                ret = ret * (zi - zj);
            }
        }
        ret
    }

    let degree = z.len();
    const MAX_ITER: usize = 10000;
    let tol = T::from(1e-12).unwrap();

    for _ in 0..MAX_ITER {
        let mut converged = true;
        let mut new_z = z.clone();

        for i in 0..degree {
            let h  = mapping_h(monic_coeffs, z[i]);
            let dh = mapping_dh(&z, i);
            if dh == Complex::zero() { continue; }
            let delta = h / dh;
            new_z[i] = z[i] - delta;
            let scale = z[i].norm().max(T::one()); // absolute near the origin, relative far away
            if delta.norm() > tol * scale { converged = false; }
        }
        z = new_z;

        if converged { return z; }
    }
    z
}
