//! Discrete-time signal processing elements, updated once per sample, and excitation signals.

mod delayer;
pub use delayer::Delayer;

mod differentiator;
pub use differentiator::Differentiator;

pub mod excitation;
