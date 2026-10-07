//! Digitalservo Motion Control Library: building blocks for the design and analysis of motion control systems.

mod analysis;
pub use analysis::*;

pub mod logger;

pub mod discretize;

mod system;
pub use system::*;

mod math;
pub use math::*;

pub mod laplace_transform;

pub mod trajectory;

pub mod feedforward;

pub mod signal;

mod sampling;

pub mod system_identification;

pub mod status;

// Compile and run the code blocks of README.md as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
