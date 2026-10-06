//! Continuous-time to discrete-time conversion.
//!
//! The method is a value passed to `discretize`, so every method is called the same way, either as
//! its own type (`Zoh`, `Tustin`, `MatchedZ`, with the method's own error type) or chosen at run
//! time with `DiscretizeMethod` (with `DiscretizeError`):
//!
//! ```
//! use dsmc::{tf, StateSpace};
//! use dsmc::discretize::{DiscretizeMethod, MatchedZ, Tustin, Zoh, ZerosAtInfinity};
//!
//! let g = tf!("100 / (s + 100)");
//! let ss = StateSpace::try_from(&g).unwrap();
//! let ts = 1e-3;
//!
//! let g_zoh = g.discretize(Zoh, ts).unwrap();
//! let g_tustin = g.discretize(Tustin, ts).unwrap();
//! let g_matched = g.discretize(MatchedZ(ZerosAtInfinity::MinusOne), ts).unwrap();
//! let ss_tustin = ss.discretize(Tustin, ts).unwrap();
//!
//! for method in [DiscretizeMethod::Zoh, DiscretizeMethod::Tustin, DiscretizeMethod::MatchedZ(ZerosAtInfinity::MinusOne)] {
//!     let g_z = g.discretize(method, ts).unwrap();
//!     let ss_z = ss.discretize(method, ts).unwrap();
//! }
//! ```
//!
//! | Method | `TransferFunction` | `StateSpace` | `ContinuousWithDelay` |
//! | --- | --- | --- | --- |
//! | `Zoh` | yes | yes (any inputs / outputs) | no |
//! | `Tustin` | yes | yes (any inputs / outputs) | no |
//! | `MatchedZ` | yes | yes (single input / output, through the transfer function) | yes |
//!
//! Using a method on a kind of system it does not support is a compile error.

pub mod bilinear_transform;
pub mod matched_z_transform;
pub mod zoh;

pub use bilinear_transform::Tustin;
pub use matched_z_transform::{ContinuousWithDelay, MatchedZ, ZerosAtInfinity};
pub use zoh::Zoh;
pub use matched_z_transform::MatchedZError;

use std::convert::Infallible;

use thiserror::Error;

use crate::{Continuous, StateSpace, StateSpaceError, TransferFunction};

/// A discretization method for continuous-time systems of type `S` with sampling period of type `T`
/// (`Zoh`, `Tustin`, `MatchedZ`, `DiscretizeMethod`). Called through `discretize` of the system.
pub trait Method<T, S> {
    /// The discrete-time system.
    type Output;
    /// Why the discretization can fail.
    type Error;

    /// Discretize `system` with sampling period `ts`.
    fn apply(&self, system: &S, ts: T) -> Result<Self::Output, Self::Error>;
}

/// Discretization method chosen at run time; works on every kind of system all of `Zoh`,
/// `Tustin` and `MatchedZ` support (`TransferFunction`, `StateSpace`), with the same result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscretizeMethod {
    /// Zero-order hold (`Zoh`).
    Zoh,
    /// Bilinear (Tustin) transform (`Tustin`).
    Tustin,
    /// Matched z-transform (`MatchedZ`).
    MatchedZ(ZerosAtInfinity),
}

/// Errors of `DiscretizeMethod`: those of the method used.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DiscretizeError {
    /// From `Zoh` or `Tustin`.
    #[error(transparent)]
    StateSpace(#[from] StateSpaceError),
    /// From `MatchedZ`.
    #[error(transparent)]
    MatchedZ(#[from] MatchedZError),
}

impl From<Infallible> for DiscretizeError {
    fn from(never: Infallible) -> Self {
        match never {}
    }
}

impl<T, S, O> Method<T, S> for DiscretizeMethod
where
    Zoh: Method<T, S, Output = O>,
    Tustin: Method<T, S, Output = O>,
    MatchedZ: Method<T, S, Output = O>,
    DiscretizeError: From<<Zoh as Method<T, S>>::Error> + From<<Tustin as Method<T, S>>::Error> + From<<MatchedZ as Method<T, S>>::Error>,
{
    type Output = O;
    type Error = DiscretizeError;

    fn apply(&self, system: &S, ts: T) -> Result<O, DiscretizeError> {
        match *self {
            Self::Zoh => Ok(Zoh.apply(system, ts)?),
            Self::Tustin => Ok(Tustin.apply(system, ts)?),
            Self::MatchedZ(zeros_at_infinity) => Ok(MatchedZ(zeros_at_infinity).apply(system, ts)?),
        }
    }
}

impl<T> TransferFunction<T, Continuous> {
    /// Discrete-time transfer function by `method` (`Zoh`, `Tustin`, `MatchedZ`, `DiscretizeMethod`) with sampling period `ts`.
    pub fn discretize<M: Method<T, Self>>(&self, method: M, ts: T) -> Result<M::Output, M::Error> {
        method.apply(self, ts)
    }
}

impl<T> StateSpace<T, Continuous> {
    /// Discrete-time state-space model by `method` (`Zoh`, `Tustin`, `MatchedZ`, `DiscretizeMethod`) with sampling period `ts`.
    pub fn discretize<M: Method<T, Self>>(&self, method: M, ts: T) -> Result<M::Output, M::Error> {
        method.apply(self, ts)
    }
}

impl<T> ContinuousWithDelay<T> {
    /// Discrete-time transfer function by `method` (`MatchedZ`) with sampling period `ts`.
    pub fn discretize<M: Method<T, Self>>(&self, method: M, ts: T) -> Result<M::Output, M::Error> {
        method.apply(self, ts)
    }
}
