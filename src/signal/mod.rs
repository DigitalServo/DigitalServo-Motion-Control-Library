//! Discrete-time signal processing elements, updated once per sample.

mod delayer;
pub use delayer::Delayer;

mod differentiator;
pub use differentiator::Differentiator;
