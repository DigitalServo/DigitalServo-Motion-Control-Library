//! Discrete-time signal processing elements, updated once per sample, excitation signals, and the
//! detection of the period of a periodic signal.

mod delayer;
pub use delayer::Delayer;

mod differentiator;
pub use differentiator::Differentiator;

pub mod excitation;

mod period;
pub use period::{detect_period, PERIOD_CORRELATION_THRESHOLD, PERIOD_PEAK_FRACTION};
