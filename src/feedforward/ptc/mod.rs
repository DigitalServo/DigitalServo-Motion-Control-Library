//! Multirate perfect tracking control (PTC): lifted model, input calculation, and state
//! references by stable inversion (bilateral Laplace transform) for output references.

mod lifted;
pub use lifted::{LiftedDiscretizedSystem, PtcError};

mod state_reference;
pub use state_reference::{ReferenceSignal, StateReference};
