//! Time-domain markers shared by `TransferFunction` and `StateSpace`.

/// Marker for continuous-time systems (polynomials in `s`, `dx/dt = Ax + Bu`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Continuous;

/// Marker for discrete-time systems (polynomials in `z`, `x[k+1] = Ax[k] + Bu[k]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Discrete;

/// Time domain of a `TransferFunction` / `PzMap` / `StateSpace`: `Continuous` or `Discrete`.
pub trait Domain: Clone + Copy + std::fmt::Debug {
    /// Variable name used by the parser and `Display` (`s` or `z`).
    const VARIABLE: char;
}

impl Domain for Continuous {
    const VARIABLE: char = 's';
}

impl Domain for Discrete {
    const VARIABLE: char = 'z';
}
