mod domain;
mod transfer_function;
mod state_space_representation;

pub use domain::{Continuous, Discrete, Domain};
pub use transfer_function::{PartialFraction, PoleTerm, PzMap, TransferFunction, TransferFunctionParseError};
pub(crate) use transfer_function::{fmt_num, principal_part, push_term};
#[doc(hidden)]
pub use transfer_function::{__detect_domain, __DomainTag, __SelectDomain};
pub use state_space_representation::{StateSpace, StateSpaceError, StateSpaceOrder};
