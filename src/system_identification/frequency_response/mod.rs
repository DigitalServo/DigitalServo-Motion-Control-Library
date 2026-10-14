//! Identification of a continuous-time transfer function `G(s)` from frequency-response samples,
//! and of `G(s)` or `G(z)` from the input / output spectra of periodic data (`sml`).

pub mod levy;
pub mod sml;
pub mod vector_fitting;
