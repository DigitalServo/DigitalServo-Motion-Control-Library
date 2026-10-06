//! System descriptor: transfer function and state-space-representation.

mod domain;
mod transfer_function;
mod state_space_representation;
mod discrete_system;

pub use domain::{Continuous, Discrete, Domain};
pub use transfer_function::{PartialFraction, PoleTerm, PzMap, TransferFunction, TransferFunctionParseError, TransferFunctionWithDelay, SimulationError};
pub(crate) use transfer_function::{fmt_num, principal_part, push_term, roots_with_multiplicity};
#[doc(hidden)]
pub use transfer_function::{__detect_domain, __DomainTag, __SelectDomain};
pub use state_space_representation::{StateSpace, StateSpaceError, StateSpaceOrder};
pub use discrete_system::{DiscreteSystem, Mimo, Siso};
