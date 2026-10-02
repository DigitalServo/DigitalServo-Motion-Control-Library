//! Multirate perfect tracking control (PTC).

mod lifted;
pub use lifted::{LiftedDiscretizedSystem, PtcError};

mod state_reference;
pub use state_reference::{ReferenceSignal, StateReference};
