mod domain;
mod transfer_function;
mod state_space_representation;

pub use domain::{Continuous, Discrete, Domain};
pub use transfer_function::{
    DelayedRational, LaplaceSignal, PartialFraction, PiecewisePolynomial, PoleTerm, PzMap, ReferenceSignal,
    StableInverse, StableInverseError, StableInverseTimeDomain, StateReference, TimeDomain, TransferFunction,
    TransferFunctionParseError,
};
#[doc(hidden)]
pub use transfer_function::{__detect_domain, __DomainTag, __SelectDomain};
pub use state_space_representation::{StateSpace, StateSpaceError, StateSpaceOrder};
