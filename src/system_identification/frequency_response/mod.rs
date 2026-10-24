//! Identification of a continuous-time transfer function `G(s)` from frequency-response samples,
//! and of `G(s)` or `G(z)` from the input / output spectra of periodic data (`sml`).

pub mod levy;
pub mod sml;
pub mod vector_fitting;

use nalgebra::{linalg::SVD, DMatrix, RealField};

/// SVD of `m` with the iterations bounded by `6 k^2` (`k = min(rows, columns)`, the limit of
/// LAPACK's `dbdsqr`), or `None` if `m` is not finite or the SVD does not converge: nalgebra's
/// `svd` iterates without limit, forever on a matrix with NaN.
pub(crate) fn bounded_svd<T: RealField + Copy>(m: DMatrix<T>, compute_u: bool, compute_v: bool) -> Option<SVD<T, nalgebra::Dyn, nalgebra::Dyn>> {
    if m.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let k = m.nrows().min(m.ncols());
    m.try_svd(compute_u, compute_v, T::default_epsilon(), 6 * k * k)
}
